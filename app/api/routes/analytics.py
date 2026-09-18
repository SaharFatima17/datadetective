from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.deps import authorize_dataset, get_current_user, require_role
from app.database import get_db
from app.models import Brief, Chart, Document, ToolRun, User
from app.schemas.requests import (
    AnalyzeRequest,
    ChartRequest,
    DocumentIndexRequest,
    ForecastRequest,
    SearchRequest,
    SQLQueryRequest,
    StatisticalTestRequest,
    ToolCallRequest,
)
from app.services import rag
from app.tools import registry

router = APIRouter(prefix="/api", tags=["analytics"])


def _authorize(db, dataset_id, user):
    """Dataset-scoped analytics routes must not read another user's data."""
    authorize_dataset(db, dataset_id, user)


def _run(db, tool, params):
    result, run = registry.call_tool(db, tool, params, agent_name="api")
    db.commit()
    if run.status == "error":
        raise HTTPException(400, run.error_message)
    return {"tool_run_id": str(run.id), "duration_ms": run.duration_ms, **result}


# ===================== Phase 5: analytics ============================ #
@router.post("/datasets/{dataset_id}/analyze")
def analyze(dataset_id: uuid.UUID, payload: AnalyzeRequest,
            user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Whitelisted dataframe operations only - never arbitrary code (proposal Sec.19)."""
    _authorize(db, dataset_id, user)
    return _run(db, "run_dataframe_code", {
        "dataset_id": str(dataset_id), "operation": payload.operation,
        "params": payload.params, "version_id": payload.version_id,
    })


@router.post("/datasets/{dataset_id}/query-sql")
def query_sql(dataset_id: uuid.UUID, payload: SQLQueryRequest,
              user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _authorize(db, dataset_id, user)
    return _run(db, "run_readonly_sql", {
        "dataset_id": str(dataset_id), "query": payload.query, "version_id": payload.version_id,
    })


@router.post("/datasets/{dataset_id}/statistical-test")
def statistical_test(dataset_id: uuid.UUID, payload: StatisticalTestRequest,
                     user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    _authorize(db, dataset_id, user)
    return _run(db, "run_statistical_test", {
        "dataset_id": str(dataset_id), "test": payload.test,
        "params": payload.params, "version_id": payload.version_id,
    })


@router.post("/datasets/{dataset_id}/chart")
def chart(dataset_id: uuid.UUID, payload: ChartRequest,
          user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _authorize(db, dataset_id, user)
    return _run(db, "render_chart", {
        "dataset_id": str(dataset_id), "spec": payload.spec,
        "investigation_id": payload.investigation_id, "version_id": payload.version_id,
    })


@router.get("/charts/{chart_id}")
def get_chart(chart_id: uuid.UUID, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    chart_row = db.get(Chart, chart_id)
    if not chart_row or not chart_row.storage_path or not Path(chart_row.storage_path).exists():
        raise HTTPException(404, "Chart not found")
    return FileResponse(chart_row.storage_path, media_type="image/png")


# ===================== Phase 9: forecasting ========================== #
@router.post("/datasets/{dataset_id}/forecast")
def forecast(dataset_id: uuid.UUID, payload: ForecastRequest,
             user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Backtested forecast with confidence bands, or a withheld explanation."""
    _authorize(db, dataset_id, user)
    return _run(db, "run_forecast", {
        "dataset_id": str(dataset_id), "date_column": payload.date_column,
        "metric": payload.metric, "horizon": payload.horizon, "freq": payload.freq,
        "agg": payload.agg, "filters": payload.filters, "version_id": payload.version_id,
    })


# ===================== Phase 6: RAG ================================== #
@router.post("/documents")
def index_document(payload: DocumentIndexRequest,
                   user: User = Depends(require_role("admin", "analyst")),
                   db: Session = Depends(get_db)):
    doc = rag.index_document(db, owner_id=user.id, title=payload.title, text=payload.text,
                             document_type=payload.document_type, metadata=payload.metadata)
    db.commit()
    return {"document_id": str(doc.id), "title": doc.title,
            "document_type": doc.document_type, "chunks": len(doc.chunks)}


@router.get("/documents")
def list_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # This had no filter at all: every account saw every document, including
    # reports written for other people's investigations.
    query = db.query(Document)
    if user.role != "admin":
        query = query.filter(Document.owner_id == user.id)
    rows = query.all()
    return {"count": len(rows), "documents": [
        {"id": str(d.id), "title": d.title, "type": d.document_type,
         "status": d.status, "chunks": len(d.chunks), "created_at": d.created_at}
        for d in rows
    ]}


@router.delete("/documents/{document_id}")
def delete_document(document_id: uuid.UUID,
                    user: User = Depends(require_role("admin", "analyst")),
                    db: Session = Depends(get_db)):
    """Remove a document from retrieval.

    A page fetched by mistake stays in the agents' context for every future
    investigation, quietly shaping plans with something irrelevant. There has
    to be a way to take it back out.

    The deletion is real: the chunks and their vectors go too, because a
    soft-deleted chunk would still be returned by a similarity search. The
    stored snapshot under the original source is left alone — that is the
    provenance record of what was retrieved, and findings already made may
    still point at it.
    """
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "No such document")
    if doc.owner_id and doc.owner_id != user.id and user.role != "admin":
        raise HTTPException(403, "This document belongs to another user")

    title, chunks = doc.title, len(doc.chunks)
    db.delete(doc)          # chunks cascade with it
    db.commit()
    return {"deleted": True, "title": title, "chunks_removed": chunks}


@router.post("/search")
def search(payload: SearchRequest, user: User = Depends(get_current_user),
           db: Session = Depends(get_db)):
    # Scoped to the caller, like every other listing. An administrator sees
    # everything, which is what makes the evaluation data reachable.
    return {"query": payload.query,
            "results": rag.search(
                db, payload.query, top_k=payload.top_k,
                document_type=payload.document_type,
                owner_id=None if user.role == "admin" else user.id)}


@router.post("/ask")
def ask_documents(payload: SearchRequest, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """Answer a question from the knowledge base alone — no dataset required.

    Some questions are not about a metric at all. "What does this company do?"
    is answered by the pages that were indexed, not by a statistical test, and
    demanding a spreadsheet first would make the knowledge base decorative.
    """
    return rag.answer_from_documents(
        db, payload.query, top_k=payload.top_k or 6,
        owner_id=None if user.role == "admin" else user.id)


@router.post("/brief")
def brief(payload: SearchRequest,
          user: User = Depends(require_role("admin", "analyst")),
          db: Session = Depends(get_db)):
    """Compose a cited brief from the knowledge base, and keep it.

    Deliberately not filed under Reports: that page promises a report for every
    completed investigation, and a document brief is a different claim. Mixing
    them would blur the one distinction this system exists to hold.

    The result is stored as written. Reopening it later shows what it said when
    it was made, not what the same question would return today after documents
    have been added or removed — otherwise a brief someone acted on could
    quietly change underneath them.
    """
    result = rag.compose_brief(
        db, payload.query, top_k=payload.top_k or 14,
        owner_id=None if user.role == "admin" else user.id)
    if not result.get("available"):
        return result

    row = Brief(
        owner_id=user.id,
        topic=result["topic"][:500],
        title=result["title"][:500],
        summary=result.get("summary"),
        sections=result.get("sections"),
        sources=result.get("sources"),
        composed=bool(result.get("composed")),
        basis=result.get("basis"),
    )
    db.add(row)
    db.commit()
    return {**result, "id": str(row.id), "created_at": row.created_at}


def _brief_payload(row: Brief) -> dict:
    return {
        "available": True,
        "id": str(row.id),
        "topic": row.topic,
        "title": row.title,
        "summary": row.summary,
        "sections": row.sections or [],
        "sources": row.sources or [],
        "composed": row.composed,
        "basis": row.basis,
        "created_at": row.created_at,
    }


@router.get("/briefs")
def list_briefs(user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    query = db.query(Brief)
    if user.role != "admin":
        query = query.filter(Brief.owner_id == user.id)
    rows = query.order_by(Brief.created_at.desc()).all()
    return {
        "count": len(rows),
        "briefs": [
            {"id": str(r.id), "title": r.title, "topic": r.topic,
             "summary": (r.summary or "")[:280], "composed": r.composed,
             "sections": len(r.sections or []), "sources": len(r.sources or []),
             "created_at": r.created_at}
            for r in rows
        ],
    }


def _own_brief(db: Session, brief_id: uuid.UUID, user: User) -> Brief:
    row = db.get(Brief, brief_id)
    if not row:
        raise HTTPException(404, "No such brief")
    if row.owner_id and row.owner_id != user.id and user.role != "admin":
        raise HTTPException(403, "That brief belongs to another user")
    return row


@router.get("/briefs/{brief_id}")
def get_brief(brief_id: uuid.UUID, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    return _brief_payload(_own_brief(db, brief_id, user))


@router.delete("/briefs/{brief_id}")
def delete_brief(brief_id: uuid.UUID,
                 user: User = Depends(require_role("admin", "analyst")),
                 db: Session = Depends(get_db)):
    row = _own_brief(db, brief_id, user)
    title = row.title
    db.delete(row)
    db.commit()
    return {"deleted": True, "title": title}


@router.get("/definitions/{term}")
def definition(term: str, user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    return rag.get_business_definition(
        db, term, owner_id=None if user.role == "admin" else user.id)


# ===================== tool layer (Phase 8 surface) ================== #
@router.get("/tools")
def list_tools(user: User = Depends(get_current_user)):
    """The same registry the MCP server exposes (proposal Sec.9)."""
    return {"tools": [{"name": n, "description": d}
                      for n, d in registry.TOOL_DESCRIPTIONS.items()]}


@router.post("/tools/call")
def call_tool(payload: ToolCallRequest,
              user: User = Depends(require_role("admin", "analyst")),
              db: Session = Depends(get_db)):
    target = payload.params.get("dataset_id")
    if target:
        _authorize(db, uuid.UUID(str(target)), user)
    return _run(db, payload.tool_name, payload.params)


@router.get("/tool-runs/{run_id}/verify")
def verify_run(run_id: uuid.UUID, user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    """Re-executes a recorded run and compares checksums (proposal Sec.20)."""
    run = db.get(ToolRun, run_id)
    if not run:
        raise HTTPException(404, "Tool run not found")
    result = registry.reverify(db, run)
    db.commit()
    return result