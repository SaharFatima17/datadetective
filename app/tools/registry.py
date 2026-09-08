"""The tool layer (proposal Sec.9).

Every analytical capability an agent can reach lives here, behind one function:
`call_tool`. That function writes a tool_runs row for every invocation, including
a checksum of the result.

That checksum is what makes verification real: the Verifier re-runs the recorded
tool with the recorded parameters and compares checksums, instead of asking an
LLM to check its own arithmetic.

Phase 8 exposes exactly this registry over MCP (see mcp_server.py) - the agents
never learn a second interface.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Chart,
    DataSource,
    Dataset,
    DatasetVersion,
    Document,
    Finding,
    Investigation,
    MissingEvidenceRequest,
    Recommendation,
    Report,
    ToolRun,
)
from app.services import (
    analytics,
    cleaning,
    dataquality,
    forecasting,
    ingestion,
    profiling,
    rag,
)


def _checksum(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _version(db: Session, dataset_id: str, version_id: str | None = None) -> DatasetVersion:
    if version_id:
        v = db.get(DatasetVersion, uuid.UUID(str(version_id)))
        if v:
            return v
    dataset = db.get(Dataset, uuid.UUID(str(dataset_id)))
    if not dataset:
        raise ValueError(f"Dataset not found: {dataset_id}")
    version = db.get(DatasetVersion, dataset.current_version_id)
    if not version:
        raise ValueError("Dataset has no version")
    return version


# --------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------- #
def inspect_dataset(db: Session, dataset_id: str, version_id: str | None = None) -> dict:
    version = _version(db, dataset_id, version_id)
    df = ingestion.load_version(version)
    return {
        "dataset_id": str(version.dataset_id),
        "version_id": str(version.id),
        "version_number": version.version_number,
        "version_type": version.version_type,
        "rows": len(df),
        "columns": [
            {"name": str(c), "dtype": str(df[c].dtype),
             "inferred_type": profiling.infer_type(df[c], str(c))}
            for c in df.columns
        ],
        "sample": df.head(5).astype(object).where(df.head(5).notna(), None).to_dict(orient="records"),
    }


def profile_columns(db: Session, dataset_id: str, version_id: str | None = None) -> dict:
    version = _version(db, dataset_id, version_id)
    df = ingestion.load_version(version)
    return profiling.profile_version(db, version, df)


def run_dataframe_code(db: Session, dataset_id: str, operation: str,
                       params: dict | None = None, version_id: str | None = None) -> dict:
    version = _version(db, dataset_id, version_id)
    df = ingestion.load_version(version)
    return analytics.run_dataframe_op(df, operation, params or {})


def run_readonly_sql(db: Session, dataset_id: str, query: str,
                     version_id: str | None = None) -> dict:
    version = _version(db, dataset_id, version_id)
    return analytics.run_readonly_sql(version.storage_path, query)


def run_statistical_test(db: Session, dataset_id: str, test: str,
                         params: dict | None = None, version_id: str | None = None) -> dict:
    version = _version(db, dataset_id, version_id)
    df = ingestion.load_version(version)
    return analytics.statistical_test(df, test, params or {})


def render_chart(db: Session, dataset_id: str, spec: dict,
                 investigation_id: str | None = None,
                 version_id: str | None = None) -> dict:
    version = _version(db, dataset_id, version_id)
    df = ingestion.load_version(version)
    out_dir = Path(settings.STORAGE_DIR) / "charts"
    path = analytics.render_chart(df, spec, out_dir)

    chart = Chart(
        investigation_id=uuid.UUID(investigation_id) if investigation_id else None,
        dataset_version_id=version.id,
        chart_type=spec.get("type", "bar"),
        title=spec.get("title"),
        spec=spec,
        storage_path=path,
    )
    db.add(chart)
    db.flush()
    return {"chart_id": str(chart.id), "storage_path": path, "type": chart.chart_type}


def run_forecast(db: Session, dataset_id: str, date_column: str, metric: str,
                 horizon: int = 3, freq: str = "M", agg: str = "sum",
                 filters: dict | None = None, version_id: str | None = None,
                 external_features: list[str] | None = None,
                 use_external_features: bool = True) -> dict:
    """Fit and backtest a forecast for one metric (proposal Sec.12).

    Other numeric columns (price, stock levels, promotions) are passed as
    external features when present, which is the condition Sec.12 gives for
    selecting a gradient-boosted regressor over a pure time-series model.
    """
    import pandas as _pd

    version = _version(db, dataset_id, version_id)
    df = ingestion.load_version(version)
    if filters:
        for col, val in filters.items():
            df = df[df[col].astype(str) == str(val)]

    series = forecasting.build_series(df, date_column, metric, freq=freq, agg=agg)

    exog = None
    used: list[str] = []
    if use_external_features:
        numeric = [
            str(c) for c in df.select_dtypes(include="number").columns
            if str(c) != metric
        ]
        candidates = external_features or numeric
        candidates = [c for c in candidates if c in df.columns and c != metric][:4]
        if candidates and len(series) > 0:
            frame = df[[date_column] + candidates].copy()
            frame[date_column] = profiling.parse_datetimes(frame[date_column])
            frame = frame.dropna(subset=[date_column])
            resampled = (frame.set_index(date_column)[candidates]
                         .resample(analytics.normalize_freq(freq)).mean())
            aligned = resampled.reindex(series.index)
            if not aligned.isna().all().all():
                exog = aligned.ffill().bfill()
                used = candidates

    result = forecasting.forecast_series(series, horizon=horizon, freq=freq, exog=exog)
    result["segment"] = filters
    result["target_metric"] = metric
    result["external_features_used"] = used
    return result


def vector_search(db: Session, query: str, top_k: int = 5,
                  document_type: str | None = None) -> dict:
    return {"query": query, "results": rag.search(db, query, top_k=top_k,
                                                  document_type=document_type)}


def get_business_definition(db: Session, term: str) -> dict:
    result = rag.get_business_definition(db, term)
    return result or {"term": term, "definition": None,
                      "note": "No definition found in the indexed knowledge base."}


def list_sources(db: Session) -> dict:
    rows = db.query(DataSource).filter(DataSource.is_deleted.is_(False)).all()
    return {
        "count": len(rows),
        "sources": [
            {"id": str(s.id), "name": s.name, "type": s.source_type,
             "format": s.source_format, "status": s.status}
            for s in rows
        ],
    }




# --------------------------------------------------------------------- #
# Source ingestion (proposal Sec.9: ingest_source / extract_document / fetch_url)
# --------------------------------------------------------------------- #
def ingest_source(db: Session, filename: str, content_base64: str,
                  owner_id: str | None = None) -> dict:
    """Register a file, preserving the original and its provenance.

    Content is passed base64-encoded rather than as a filesystem path, so an
    agent cannot use this tool to read arbitrary files off the machine.
    """
    import base64
    from pathlib import Path as _Path

    try:
        content = base64.b64decode(content_base64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"content_base64 is not valid base64: {exc}") from exc

    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > limit:
        raise ValueError(f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit")

    ext = _Path(filename).suffix.lower()
    if ext not in ingestion.TABULAR_EXTS | ingestion.DOCUMENT_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    owner = uuid.UUID(owner_id) if owner_id else None
    source = ingestion.register_file_source(db, filename, content, owner_id=owner)
    result = {"source_id": str(source.id), "name": source.name,
              "checksum": source.checksum_sha256, "size_bytes": source.size_bytes}

    if ext in ingestion.TABULAR_EXTS:
        dataset = ingestion.create_dataset_from_source(db, source)
        version = db.get(DatasetVersion, dataset.current_version_id)
        profiling.profile_version(db, version, ingestion.load_version(version))
        result.update(kind="dataset", dataset_id=str(dataset.id),
                      version_id=str(version.id))
    else:
        result.update(kind="document",
                      note="Call extract_document to index its text for retrieval.")
    return result


def extract_document(db: Session, source_id: str,
                     document_type: str = "business_doc") -> dict:
    """Extract text from a registered document and index it for retrieval."""
    from pathlib import Path as _Path

    source = db.get(DataSource, uuid.UUID(str(source_id)))
    if not source:
        raise ValueError(f"Source not found: {source_id}")
    artifact = next((a for a in source.artifacts
                     if a.artifact_type in {"original", "snapshot"}), None)
    if not artifact:
        raise ValueError("This source has no stored original to extract from")

    text = ingestion.extract_text(_Path(artifact.storage_path))
    doc = rag.index_document(db, title=source.name, text=text,
                             document_type=document_type, source_id=source.id)
    source.status = "extracted"
    db.flush()
    return {"document_id": str(doc.id), "title": doc.title,
            "characters": len(text), "chunks": len(doc.chunks)}


def fetch_url(db: Session, url: str, index_for_retrieval: bool = True) -> dict:
    """Retrieve permitted web content, storing a snapshot with provenance."""
    from pathlib import Path as _Path

    source = ingestion.fetch_url(db, url)
    result = {"source_id": str(source.id), "url": url,
              "checksum": source.checksum_sha256}
    if index_for_retrieval:
        text = ingestion.extract_text(_Path(source.artifacts[0].storage_path))
        doc = rag.index_document(db, title=url, text=text,
                                 document_type="web_page", source_id=source.id)
        source.status = "extracted"
        result.update(document_id=str(doc.id), chunks=len(doc.chunks))
    db.flush()
    return result


def detect_drift(db: Session, dataset_id: str, from_version: str | None = None,
                 to_version: str | None = None) -> dict:
    """Compare two dataset versions for distribution drift (proposal Sec.10)."""
    dataset = db.get(Dataset, uuid.UUID(str(dataset_id)))
    if not dataset:
        raise ValueError(f"Dataset not found: {dataset_id}")
    ordered = sorted(dataset.versions, key=lambda v: v.version_number)
    if len(ordered) < 2 and not (from_version and to_version):
        raise ValueError("This dataset has only one version to compare")

    later = _version(db, dataset_id, to_version) if to_version else ordered[-1]
    earlier = _version(db, dataset_id, from_version) if from_version else ordered[-2]
    return {
        "from_version": earlier.version_number,
        "to_version": later.version_number,
        **dataquality.detect_drift(ingestion.load_version(earlier),
                                   ingestion.load_version(later)),
    }


# --------------------------------------------------------------------- #
# Cleaning (proposal Sec.9: apply_cleaning_plan)
# --------------------------------------------------------------------- #
def apply_cleaning_plan(db: Session, dataset_id: str,
                        approved_op_ids: list[str] | None = None,
                        version_id: str | None = None) -> dict:
    """Apply a cleaning plan, creating a NEW version (proposal Sec.10).

    Safe operations always run; destructive ones only when their op_id appears
    in approved_op_ids. The parent version is never modified.
    """
    dataset = db.get(Dataset, uuid.UUID(str(dataset_id)))
    if not dataset:
        raise ValueError(f"Dataset not found: {dataset_id}")
    parent = _version(db, dataset_id, version_id)
    df = ingestion.load_version(parent)

    plan = cleaning.propose_plan(df)
    cleaned, log = cleaning.apply_plan(df, plan, approved_op_ids or [])
    version = ingestion.new_version(
        db, dataset, cleaned, parent, version_type="cleaned",
        cleaning_operations={"operations": log, "parent_version": str(parent.id)},
        created_by_agent="cleaning",
    )
    after = profiling.profile_version(db, version, cleaned)
    return {
        "new_version_id": str(version.id),
        "version_number": version.version_number,
        "executed": [o for o in log if o.get("executed")],
        "skipped": [o for o in log if not o.get("executed")],
        "health_before": (parent.profile_summary or {}).get("health_score"),
        "health_after": after["health_score"],
    }


# --------------------------------------------------------------------- #
# Persistence (proposal Sec.9: save_finding / save_recommendation / save_report)
# --------------------------------------------------------------------- #
def save_finding(db: Session, investigation_id: str, statement: str,
                 evidence_summary: str | None = None, tool_run_id: str | None = None,
                 finding_type: str = "measurement", magnitude: float | None = None,
                 unit: str | None = None, confidence: str = "medium",
                 caveats: str | None = None) -> dict:
    """Persist a finding. A finding without a tool_run_id cannot be verified."""
    if finding_type not in {"measurement", "association", "driver"}:
        raise ValueError("finding_type must be measurement, association or driver")

    finding = Finding(
        investigation_id=uuid.UUID(str(investigation_id)),
        tool_run_id=uuid.UUID(str(tool_run_id)) if tool_run_id else None,
        statement=statement,
        finding_type=finding_type,
        evidence_summary=evidence_summary,
        magnitude=magnitude,
        unit=unit,
        confidence=confidence,
        caveats=caveats,
    )
    db.add(finding)
    db.flush()
    return {
        "finding_id": str(finding.id),
        "verification_status": finding.verification_status,
        "note": ("No tool run was linked, so this finding cannot pass verification."
                 if not tool_run_id else "Verify with tool_runs/{id}/verify."),
    }


def generate_recommendations(db: Session, investigation_id: str) -> dict:
    """Turn verified driver findings into ranked actions (proposal Sec.13)."""
    from app.services import recommendations as rec_service

    investigation = db.get(Investigation, uuid.UUID(str(investigation_id)))
    if not investigation:
        raise ValueError(f"Investigation not found: {investigation_id}")

    findings = (db.query(Finding)
                .filter(Finding.investigation_id == investigation.id).all())
    forecasts = {}
    from app.models import Forecast

    for forecast in (db.query(Forecast)
                     .filter(Forecast.investigation_id == investigation.id).all()):
        if forecast.finding_id:
            forecasts[str(forecast.finding_id)] = forecast

    recs = rec_service.generate_recommendations(db, investigation, findings, forecasts)
    return {
        "count": len(recs),
        "recommendations": [
            {"id": str(r.id), "rank": r.rank, "action": r.action,
             "expected_impact": r.expected_impact, "impact_method": r.impact_method,
             "confidence": r.confidence, "urgency": r.urgency}
            for r in recs
        ],
    }


def save_recommendation(db: Session, investigation_id: str, action: str,
                        finding_id: str | None = None, rationale: str | None = None,
                        expected_impact: dict | None = None,
                        impact_method: str | None = None,
                        confidence: str = "medium", urgency: str = "medium") -> dict:
    """Persist a recommendation. Rejected unless its finding is verified (Sec.13)."""
    if finding_id:
        finding = db.get(Finding, uuid.UUID(str(finding_id)))
        if not finding:
            raise ValueError(f"Finding not found: {finding_id}")
        if finding.verification_status != "verified":
            raise ValueError(
                "An unverified finding cannot produce a recommendation "
                "(proposal Sec.13)."
            )

    rec = Recommendation(
        investigation_id=uuid.UUID(str(investigation_id)),
        finding_id=uuid.UUID(str(finding_id)) if finding_id else None,
        action=action,
        rationale=rationale,
        expected_impact=expected_impact,
        impact_method=impact_method or "Supplied by caller; assumptions not recorded.",
        confidence=confidence,
        urgency=urgency,
    )
    db.add(rec)
    db.flush()
    return {"recommendation_id": str(rec.id), "action": rec.action}


def save_report(db: Session, investigation_id: str, executive_summary: str,
                title: str | None = None, content: dict | None = None,
                index_for_retrieval: bool = True) -> dict:
    """Persist a versioned report and make it retrievable later (Sec.14, Sec.18)."""
    investigation = db.get(Investigation, uuid.UUID(str(investigation_id)))
    if not investigation:
        raise ValueError(f"Investigation not found: {investigation_id}")

    existing = (db.query(Report)
                .filter(Report.investigation_id == investigation.id).count())
    report = Report(
        investigation_id=investigation.id,
        version_number=existing + 1,
        title=title or f"Investigation report: {investigation.question[:150]}",
        executive_summary=executive_summary,
        content=content or {},
        status="final",
    )
    db.add(report)
    db.flush()
    if index_for_retrieval:
        rag.index_investigation_report(db, investigation, report)
    return {"report_id": str(report.id), "version": report.version_number}


# --------------------------------------------------------------------- #
# Investigation state (proposal Sec.9: request_missing_evidence /
# resume_investigation / compare_investigations)
# --------------------------------------------------------------------- #
def request_missing_evidence(db: Session, investigation_id: str, question: str,
                             request_type: str = "clarification",
                             reason: str | None = None,
                             hypothesis_id: str | None = None) -> dict:
    """Raise a user-facing request and pause the investigation (Sec.7 step 15)."""
    allowed = {"missing_field", "missing_dataset", "missing_document",
               "definition", "clarification"}
    if request_type not in allowed:
        raise ValueError(f"request_type must be one of: {', '.join(sorted(allowed))}")

    investigation = db.get(Investigation, uuid.UUID(str(investigation_id)))
    if not investigation:
        raise ValueError(f"Investigation not found: {investigation_id}")

    req = MissingEvidenceRequest(
        investigation_id=investigation.id,
        hypothesis_id=uuid.UUID(str(hypothesis_id)) if hypothesis_id else None,
        request_type=request_type,
        question=question,
        reason=reason,
    )
    db.add(req)
    investigation.status = "awaiting_user"
    db.flush()
    return {"request_id": str(req.id), "investigation_status": investigation.status}


def resume_investigation(db: Session, investigation_id: str, request_id: str,
                         response: str) -> dict:
    """Validate a user response and continue the investigation (Sec.7 step 16)."""
    from app.agents import orchestrator

    investigation = db.get(Investigation, uuid.UUID(str(investigation_id)))
    if not investigation:
        raise ValueError(f"Investigation not found: {investigation_id}")
    return orchestrator.resume(db, investigation, uuid.UUID(str(request_id)), response)


def compare_investigations(db: Session, current_id: str,
                           previous_ids: list[str] | None = None) -> dict:
    """Compare drivers and findings across investigations (proposal Sec.14).

    Reports whether the driver changed - the case the proposal calls out, where
    a later decline has a different cause from an earlier one.
    """
    from app.models import InvestigationComparison

    current = db.get(Investigation, uuid.UUID(str(current_id)))
    if not current:
        raise ValueError(f"Investigation not found: {current_id}")

    def _drivers(inv_id) -> list[Finding]:
        return (db.query(Finding)
                .filter(Finding.investigation_id == inv_id,
                        Finding.finding_type == "driver")
                .all())

    if previous_ids:
        previous = [db.get(Investigation, uuid.UUID(str(i))) for i in previous_ids]
        previous = [p for p in previous if p]
    else:
        previous = (db.query(Investigation)
                    .filter(Investigation.dataset_id == current.dataset_id,
                            Investigation.id != current.id,
                            Investigation.status == "complete")
                    .order_by(Investigation.created_at.desc())
                    .limit(3).all())

    current_drivers = [f.statement for f in _drivers(current.id)]
    comparisons = []
    for prev in previous:
        prev_drivers = [f.statement for f in _drivers(prev.id)]
        shared = {d.split(" contributed")[0] for d in current_drivers} & {
            d.split(" contributed")[0] for d in prev_drivers}
        if not prev_drivers and not current_drivers:
            change = "stable"
        elif not current_drivers:
            change = "disappeared"
        elif not prev_drivers:
            change = "increased"
        elif shared:
            change = "stable"
        else:
            change = "replaced"

        summary = (
            f"Previous drivers: {prev_drivers or 'none'}. "
            f"Current drivers: {current_drivers or 'none'}. "
            + ("The driver has changed since the previous investigation."
               if change == "replaced" else "The same driver is still present."
               if change == "stable" else "")
        )
        record = InvestigationComparison(
            current_investigation_id=current.id,
            previous_investigation_id=prev.id,
            driver_change=change,
            comparison_summary=summary,
            details={"previous_drivers": prev_drivers,
                     "current_drivers": current_drivers},
        )
        db.add(record)
        comparisons.append({
            "previous_investigation_id": str(prev.id),
            "previous_question": prev.question,
            "driver_change": change,
            "summary": summary,
        })
    db.flush()
    return {"current_investigation_id": str(current.id),
            "current_drivers": current_drivers,
            "compared_with": len(comparisons),
            "comparisons": comparisons}


# --------------------------------------------------------------------- #
TOOLS = {
    "inspect_dataset": inspect_dataset,
    "profile_columns": profile_columns,
    "run_dataframe_code": run_dataframe_code,
    "run_readonly_sql": run_readonly_sql,
    "run_statistical_test": run_statistical_test,
    "render_chart": render_chart,
    "run_forecast": run_forecast,
    "vector_search": vector_search,
    "get_business_definition": get_business_definition,
    "list_sources": list_sources,
    # Sec.9 tools completed
    "ingest_source": ingest_source,
    "extract_document": extract_document,
    "fetch_url": fetch_url,
    "apply_cleaning_plan": apply_cleaning_plan,
    "save_finding": save_finding,
    "generate_recommendations": generate_recommendations,
    "save_recommendation": save_recommendation,
    "save_report": save_report,
    "request_missing_evidence": request_missing_evidence,
    "resume_investigation": resume_investigation,
    "compare_investigations": compare_investigations,
    "detect_drift": detect_drift,
}

TOOL_DESCRIPTIONS = {
    "inspect_dataset": "Return schema, shape and a sample of rows.",
    "profile_columns": "Compute quality and distribution statistics for every column.",
    "run_dataframe_code": "Run one whitelisted dataframe operation: describe, value_counts, groupby_aggregate, correlation, time_series.",
    "run_readonly_sql": "Run a validated SELECT query against the dataset.",
    "run_statistical_test": "Run ttest, chi_square, correlation or linear_regression with effect sizes.",
    "render_chart": "Render a bar, line, hist, scatter or forecast chart to PNG.",
    "run_forecast": "Fit and backtest a time-series model; returns point forecasts with confidence intervals.",
    "vector_search": "Semantic search across indexed documents and past investigations.",
    "get_business_definition": "Retrieve a KPI or data-dictionary definition.",
    "list_sources": "List registered data sources.",
    "ingest_source": "Register a base64-encoded file, preserving the original and its provenance.",
    "extract_document": "Extract text from a registered document and index it for retrieval.",
    "fetch_url": "Retrieve permitted web content and store a snapshot with provenance.",
    "apply_cleaning_plan": "Apply a cleaning plan, creating a new dataset version. Destructive operations need explicit approval.",
    "save_finding": "Persist a finding with its supporting tool run.",
    "generate_recommendations": "Turn verified driver findings into ranked, evidence-linked actions.",
    "save_recommendation": "Persist a recommendation. Refused if its finding is not verified.",
    "save_report": "Persist a versioned investigation report and index it for later retrieval.",
    "request_missing_evidence": "Raise a user-facing data request and pause the investigation.",
    "resume_investigation": "Validate a user response and resume a paused investigation.",
    "compare_investigations": "Compare drivers and findings against previous investigations.",
    "detect_drift": "Compare two dataset versions for distribution drift and schema changes.",
}


def call_tool(db: Session, tool_name: str, params: dict,
              investigation_id: uuid.UUID | None = None,
              hypothesis_id: uuid.UUID | None = None,
              agent_name: str | None = None) -> tuple[dict, ToolRun]:
    """Single entry point. Every call is logged, timed and checksummed."""
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {sorted(TOOLS)}")

    started = time.perf_counter()
    run = ToolRun(
        investigation_id=investigation_id,
        hypothesis_id=hypothesis_id,
        agent_name=agent_name,
        tool_name=tool_name,
        parameters=params,
    )

    try:
        result = TOOLS[tool_name](db, **params)
        run.result = json.loads(json.dumps(result, default=str))
        run.status = "success"
        run.result_checksum = _checksum(run.result)
    except Exception as exc:  # noqa: BLE001
        result = {"error": str(exc)}
        run.result = result
        run.status = "error"
        run.error_message = str(exc)

    run.duration_ms = int((time.perf_counter() - started) * 1000)
    db.add(run)
    db.flush()
    return result, run


def reverify(db: Session, run: ToolRun) -> dict:
    """Re-execute a recorded tool run and compare checksums (proposal Sec.20).

    This is what "verification" means in this system - a mechanical re-run, not
    an LLM re-reading its own output.
    """
    if run.status != "success":
        return {"verified": False, "reason": "original run did not succeed"}
    try:
        fresh = TOOLS[run.tool_name](db, **(run.parameters or {}))
    except Exception as exc:  # noqa: BLE001
        return {"verified": False, "reason": f"re-run failed: {exc}"}

    fresh_checksum = _checksum(json.loads(json.dumps(fresh, default=str)))
    return {
        "verified": fresh_checksum == run.result_checksum,
        "original_checksum": run.result_checksum,
        "recomputed_checksum": fresh_checksum,
        "tool_name": run.tool_name,
    }
