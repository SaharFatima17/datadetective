from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.agents import orchestrator
from app.core.deps import (
    authorize_dataset,
    authorize_investigation,
    get_current_user,
    require_role,
)
from app.database import get_db
from app.models import (
    Feedback,
    Finding,
    Hypothesis,
    Investigation,
    MissingEvidenceRequest,
    Recommendation,
    Report,
    ToolRun,
    User,
)
from app.config import settings
from app.models import Dataset, DatasetVersion
from app.schemas.requests import FeedbackRequest, InvestigationCreate, ResumeRequest
from app.services import ingestion

router = APIRouter(prefix="/api/investigations", tags=["investigations"])


def _get(db: Session, investigation_id: uuid.UUID, user: User) -> Investigation:
    """Fetch an investigation the caller owns (proposal Sec.19)."""
    return authorize_investigation(db, investigation_id, user)


@router.post("")
def create_and_run(payload: InvestigationCreate,
                   user: User = Depends(require_role("admin", "analyst")),
                   db: Session = Depends(get_db)):
    """Starts an investigation and runs it through to a report (proposal Sec.7)."""
    try:
        dataset = authorize_dataset(db, uuid.UUID(payload.dataset_id), user)
        inv = orchestrator.start_investigation(
            db, dataset.id, payload.question, owner_id=user.id,
            period_start=payload.period_start, period_end=payload.period_end)
        result = orchestrator.run(db, inv, forecast_horizon=payload.forecast_horizon)
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"Investigation failed: {exc}") from exc
    return result


@router.get("")
def list_investigations(user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    query = db.query(Investigation)
    if user.role != "admin":
        query = query.filter(
            (Investigation.owner_id == user.id) | (Investigation.owner_id.is_(None))
        )
    rows = query.order_by(Investigation.created_at.desc()).all()
    return {"count": len(rows), "investigations": [
        {"id": str(i.id), "question": i.question, "status": i.status,
         "target_metric": i.target_metric, "round": i.current_round, "created_at": i.created_at}
        for i in rows
    ]}


@router.get("/{investigation_id}")
def get_investigation(investigation_id: uuid.UUID, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    inv = _get(db, investigation_id, user)
    return orchestrator._state(db, inv)


@router.post("/{investigation_id}/resume")
def resume(investigation_id: uuid.UUID, payload: ResumeRequest,
           user: User = Depends(require_role("admin", "analyst")),
           db: Session = Depends(get_db)):
    """Answers a missing-evidence request and continues (proposal Sec.7 step 16)."""
    inv = _get(db, investigation_id, user)
    try:
        result = orchestrator.resume(db, inv, uuid.UUID(payload.request_id), payload.response)
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    return result


@router.post("/{investigation_id}/evidence/{request_id}/supply")
async def supply_evidence(investigation_id: uuid.UUID, request_id: uuid.UUID,
                          file: UploadFile = File(...),
                          user: User = Depends(require_role("admin", "analyst")),
                          db: Session = Depends(get_db)):
    """Answer a missing-evidence request with actual data (proposal Sec.7 step 16).

    The file is joined onto the investigation's current dataset version, creating
    an enriched version. It never becomes a separate unrelated dataset - that
    would leave the blocked hypothesis just as untestable as before.
    """
    inv = _get(db, investigation_id, user)
    req = db.get(MissingEvidenceRequest, request_id)
    if not req or req.investigation_id != inv.id:
        raise HTTPException(404, "Request not found for this investigation")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB} MB limit")

    try:
        supplementary = ingestion.parse_dataframe(file.filename, content)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc

    dataset = db.get(Dataset, inv.dataset_id)
    parent = db.get(DatasetVersion, inv.version_id)
    if not dataset or not parent:
        raise HTTPException(409, "This investigation has no dataset version to enrich")

    try:
        version = ingestion.merge_supplementary(db, dataset, parent, supplementary)
    except ValueError as exc:
        db.rollback()
        # 422: the file was readable but cannot be joined usefully
        raise HTTPException(422, str(exc)) from exc

    added = (version.cleaning_operations or {}).get("added_columns", [])
    inv.version_id = version.id
    db.flush()

    # Hand the new columns to the existing resume logic rather than repeating
    # its column-matching rules here.
    response_text = (
        f"Supplied file '{file.filename}' adds these columns to the dataset: "
        + ", ".join(added)
    )
    try:
        result = orchestrator.resume(db, inv, request_id, response_text)
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(422, str(exc)) from exc

    return result | {
        "merged_version_id": str(version.id),
        "version_number": version.version_number,
        "added_columns": added,
        "join_on": (version.cleaning_operations or {}).get("join_on"),
    }


@router.post("/{investigation_id}/recommendations/{rec_id}/feedback")
def submit_feedback(investigation_id: uuid.UUID, rec_id: uuid.UUID,
                    payload: FeedbackRequest,
                    user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """Record whether a recommendation was useful (proposal Sec.13, Sec.15).

    Feedback feeds back into later investigations: when a similar driver is
    found again, past ratings for that kind of action are surfaced.
    """
    inv = _get(db, investigation_id, user)
    rec = db.get(Recommendation, rec_id)
    if not rec or rec.investigation_id != inv.id:
        raise HTTPException(404, "Recommendation not found for this investigation")
    if payload.rating not in {"useful", "not_useful"}:
        raise HTTPException(422, "rating must be 'useful' or 'not_useful'")

    entry = Feedback(
        user_id=user.id,
        investigation_id=inv.id,
        target_type="recommendation",
        target_id=rec.id,
        rating=5 if payload.rating == "useful" else 1,
        was_adopted=payload.rating == "useful",
        comment=payload.notes,
    )
    db.add(entry)
    rec.user_decision = "accepted" if payload.rating == "useful" else "rejected"
    db.commit()
    return {"feedback_id": str(entry.id), "recommendation_id": str(rec.id),
            "rating": payload.rating, "user_decision": rec.user_decision}


@router.post("/{investigation_id}/abandon")
def abandon(investigation_id: uuid.UUID,
            user: User = Depends(require_role("admin", "analyst")),
            db: Session = Depends(get_db)):
    inv = _get(db, investigation_id, user)
    result = orchestrator.abandon(db, inv)
    db.commit()
    return result


@router.get("/{investigation_id}/report")
def get_report(investigation_id: uuid.UUID, user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    _get(db, investigation_id, user)
    report = (
        db.query(Report).filter(Report.investigation_id == investigation_id)
        .order_by(Report.version_number.desc()).first()
    )
    if not report:
        raise HTTPException(404, "No report yet for this investigation")
    return {"id": str(report.id), "version": report.version_number, "title": report.title,
            "executive_summary": report.executive_summary, **(report.content or {})}


@router.get("/{investigation_id}/timeline")
def timeline(investigation_id: uuid.UUID, user: User = Depends(get_current_user),
             db: Session = Depends(get_db)):
    """Every agent action in order - the investigation trail from Sec.17."""
    _get(db, investigation_id, user)
    runs = (
        db.query(ToolRun).filter(ToolRun.investigation_id == investigation_id)
        .order_by(ToolRun.created_at).all()
    )
    hypotheses = db.query(Hypothesis).filter(
        Hypothesis.investigation_id == investigation_id).all()
    return {
        "tool_runs": [
            {"id": str(r.id), "agent": r.agent_name, "tool": r.tool_name,
             "status": r.status, "duration_ms": r.duration_ms,
             "parameters": r.parameters, "created_at": r.created_at}
            for r in runs
        ],
        "hypotheses": [
            {"id": str(h.id), "statement": h.statement, "status": h.status,
             "confidence": h.confidence, "reasoning": h.reasoning}
            for h in hypotheses
        ],
    }


@router.get("/{investigation_id}/evidence")
def evidence(investigation_id: uuid.UUID, user: User = Depends(get_current_user),
             db: Session = Depends(get_db)):
    """Every finding with the exact tool run behind it (proposal Sec.18)."""
    _get(db, investigation_id, user)
    findings = db.query(Finding).filter(Finding.investigation_id == investigation_id).all()
    out = []
    for f in findings:
        run = db.get(ToolRun, f.tool_run_id) if f.tool_run_id else None
        out.append({
            "finding_id": str(f.id),
            "statement": f.statement,
            "evidence_summary": f.evidence_summary,
            "confidence": f.confidence,
            "verification_status": f.verification_status,
            "caveats": f.caveats,
            "finding_type": f.finding_type,
            "magnitude": f.magnitude,
            "unit": f.unit,
            # exposed so a client can re-run the exact call and compare
            # checksums, which is what verification means here
            "tool_run_id": str(f.tool_run_id) if f.tool_run_id else None,
            "calculation": {
                "tool": run.tool_name, "parameters": run.parameters,
                "result": run.result, "checksum": run.result_checksum,
            } if run else None,
        })
    return {"count": len(out), "findings": out}


@router.get("/{investigation_id}/recommendations")
def recommendations(investigation_id: uuid.UUID, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    _get(db, investigation_id, user)
    rows = (
        db.query(Recommendation).filter(Recommendation.investigation_id == investigation_id)
        .order_by(Recommendation.rank).all()
    )
    return {"count": len(rows), "recommendations": [
        {"id": str(r.id), "rank": r.rank, "action": r.action, "rationale": r.rationale,
         "expected_impact": r.expected_impact, "impact_method": r.impact_method,
         "confidence": r.confidence, "urgency": r.urgency, "user_decision": r.user_decision}
        for r in rows
    ]}


@router.get("/{investigation_id}/requests")
def open_requests(investigation_id: uuid.UUID, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    _get(db, investigation_id, user)
    rows = db.query(MissingEvidenceRequest).filter(
        MissingEvidenceRequest.investigation_id == investigation_id).all()
    return {"count": len(rows), "requests": [
        {"id": str(r.id), "type": r.request_type, "question": r.question,
         "reason": r.reason, "status": r.status, "response": r.response}
        for r in rows
    ]}