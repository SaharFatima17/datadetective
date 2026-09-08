"""The orchestrator (proposal Sec.3, Sec.7).

Runs the investigation as an explicit state machine:

    planning -> running -> awaiting_user -> running -> complete
                                         \\-> abandoned

`run` executes steps 10 through 24 of Sec.7. If the Evidence Gap Agent files a
request, the investigation stops at awaiting_user with whatever it did manage to
establish; `resume` validates the response and continues the blocked hypotheses.

max_rounds is enforced so the loop cannot run forever.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.agents import investigation_agents as agents
from app.agents import supervisor
from app.core import redaction
from app.models import (
    Dataset,
    DatasetVersion,
    Finding,
    Forecast,
    Hypothesis,
    Investigation,
    InvestigationRound,
    MissingEvidenceRequest,
    Recommendation,
    Report,
)
from app.services import ingestion, rag, recommendations
from app.tools import registry


def start_investigation(db: Session, dataset_id: uuid.UUID, question: str,
                        owner_id: uuid.UUID | None = None,
                        period_start: str | None = None,
                        period_end: str | None = None) -> Investigation:
    """Begin an investigation, optionally scoped to a date window.

    The window matters for proposal Sec.14: asking the same question about two
    different periods is what makes a CHANGE of driver visible. Without it,
    every investigation on a dataset looks at the same span and necessarily
    reaches the same conclusion.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    inv = Investigation(
        owner_id=owner_id,
        dataset_id=dataset.id,
        version_id=dataset.current_version_id,
        question=question,
        status="planning",
        comparison_period=({"scope_start": period_start, "scope_end": period_end}
                           if (period_start or period_end) else None),
    )
    db.add(inv)
    db.flush()
    db.add(InvestigationRound(investigation_id=inv.id, round_number=1, trigger="initial_question"))
    db.flush()
    return inv


# --------------------------------------------------------------------- #
# Ablation switches for the Sec.20 experiment. Turning one off isolates what
# that component contributes, which a four-way architecture comparison cannot
# show on its own because the systems differ in several ways at once.
ABLATIONS = ("critic", "verification", "rag", "forecast", "evidence_gap",
             "recommendations")


def run(db: Session, investigation: Investigation, forecast_horizon: int = 3,
        ablation: dict | None = None) -> dict:
    """Executes the investigation from planning to report.

    The whole run happens inside a redaction context, so no prompt built by any
    agent can carry a sensitive column's values off the machine (Sec.19).

    `ablation` disables named components, e.g. {"critic": False}. Unlisted
    components stay on. Used only by the evaluation harness.
    """
    if investigation.current_round > investigation.max_rounds:
        investigation.status = "abandoned"
        db.flush()
        return {"status": "abandoned", "reason": "maximum investigation rounds reached"}

    version = db.get(DatasetVersion, investigation.version_id)
    version, df = _scoped_version(db, investigation, version)

    enabled = {name: True for name in ABLATIONS}
    enabled.update({k: bool(v) for k, v in (ablation or {}).items() if k in ABLATIONS})

    with redaction.use(redaction.build(db, investigation.dataset_id, df)):
        return _run_investigation(db, investigation, version, df,
                                  forecast_horizon, enabled)


def _run_investigation(db: Session, investigation: Investigation,
                       version: DatasetVersion, df, forecast_horizon: int,
                       enabled: dict) -> dict:
    investigation.status = "running"
    db.flush()

    # --- steps 5-7: profile ------------------------------------------ #
    profile, _ = registry.call_tool(
        db, "profile_columns",
        {"dataset_id": str(investigation.dataset_id),
         "version_id": str(investigation.version_id)},
        investigation_id=investigation.id, agent_name="profiler",
    )

    # --- steps 10-12: plan ------------------------------------------- #
    # Retrieval feeds the PLAN, not just the report: business definitions and
    # uploaded documents shape which metric and dimensions get investigated
    # (proposal Sec.5 multi-source investigation, Sec.14).
    definitions = (
        supervisor.lookup_definitions(
            db, investigation.question,
            [c["name"] for c in profile.get("columns", [])])
        if enabled["rag"] else None
    )
    plan = supervisor.choose_columns(investigation.question, profile, definitions)
    plan["comparison_period"] = supervisor.split_periods(df, plan["date_column"])
    investigation.target_metric = plan["target_metric"]
    investigation.dimensions = {"columns": plan["dimensions"]}
    # merge, never overwrite: the scope window was set when the investigation
    # was created and must survive the planner writing its own split
    investigation.comparison_period = {
        **(investigation.comparison_period or {}),
        **(plan["comparison_period"] or {}),
    } or None
    investigation.plan = plan
    db.flush()

    if not plan["target_metric"]:
        investigation.status = "awaiting_user"
        req = MissingEvidenceRequest(
            investigation_id=investigation.id,
            request_type="clarification",
            question=(
                "I could not identify which column holds the metric you are asking about. "
                f"Numeric columns available: {plan['available']['numeric']}. Which one?"
            ),
            reason="No numeric column could be matched to the question.",
        )
        db.add(req)
        db.flush()
        return _state(db, investigation)

    # --- step 13: hypotheses ----------------------------------------- #
    existing = db.query(Hypothesis).filter(
        Hypothesis.investigation_id == investigation.id).all()
    if existing:
        hypotheses = existing
        # A resumed round may be working with columns that did not exist when
        # the first round ran - supplied evidence, or a merged dataset.
        hypotheses += agents.top_up_hypotheses(
            db, investigation, plan, existing, {str(c) for c in df.columns})
    else:
        hypotheses = agents.generate_hypotheses(db, investigation, plan)

    # --- steps 14-15: evidence sufficiency --------------------------- #
    pending = [h for h in hypotheses if h.status in {"proposed", "testable", "blocked_missing_evidence"}]
    if enabled["evidence_gap"]:
        requests = agents.check_evidence(db, investigation, pending, df)
    else:
        # without the gap check, every hypothesis is simply attempted
        for h in pending:
            h.status = "testable"
        db.flush()
        requests = []

    # --- step 17: test what can be tested ---------------------------- #
    for h in hypotheses:
        if h.status != "testable":
            continue
        agents.test_hypothesis(db, investigation, h, investigation.dataset_id, plan, df)

    # Read findings back from the database rather than keeping only the ones
    # created in this pass - otherwise a resumed investigation would produce a
    # report that silently drops everything established in earlier rounds.
    findings: list[Finding] = (
        db.query(Finding).filter(Finding.investigation_id == investigation.id).all()
    )

    # --- steps 18 & 21: history -------------------------------------- #
    if enabled["rag"]:
        history = agents.compare_with_history(db, investigation, findings)
        # Structured driver-change comparison against earlier investigations on
        # the same dataset (proposal Sec.14). The RAG search above finds related
        # text; this records whether the DRIVER itself changed.
        try:
            history["driver_comparison"] = registry.compare_investigations(
                db, str(investigation.id))
        except Exception as exc:  # noqa: BLE001
            history["driver_comparison"] = {"error": str(exc)}
    else:
        history = {"has_history": False, "note": "retrieval disabled"}
    agents.snapshot_metrics(db, investigation, plan, df)

    # Evidence Gap Agent, second pass (proposal Sec.11): the system can measure
    # THAT the metric moved but has no column capable of explaining WHY. That is
    # exactly the case where it must ask rather than dress a measurement up as a
    # cause.
    drivers = [f for f in findings if f.finding_type == "driver"]
    measurements = [f for f in findings if f.finding_type == "measurement"]
    already_asked = (
        db.query(MissingEvidenceRequest)
        .filter(MissingEvidenceRequest.investigation_id == investigation.id,
                MissingEvidenceRequest.request_type == "missing_dataset")
        .count()
    )
    if measurements and not drivers and not already_asked and enabled["evidence_gap"]:
        explanatory = plan["dimensions"] + [
            c for c in plan["available"]["numeric"] if c != plan["target_metric"]
        ]
        req = MissingEvidenceRequest(
            investigation_id=investigation.id,
            request_type="missing_dataset",
            question=(
                f"I can measure the change in {plan['target_metric']} "
                f"({measurements[0].statement}) but this dataset contains no column "
                "that could explain it. To find a cause I need explanatory data - for "
                "example a segment (region, product, channel, customer type) or an "
                "operational measure (price, stock availability, returns, headcount). "
                "Can you supply a dataset containing one of these?"
            ),
            reason=(
                f"Usable explanatory columns found: {explanatory or 'none'}. "
                "Without one, any stated cause would be a guess."
            ),
        )
        db.add(req)
        requests.append(req)
        db.flush()

    # --- step 19: critique ------------------------------------------- #
    critique_result = (agents.critique(db, investigation, findings)
                       if enabled["critic"]
                       else {"concerns": [], "verdict": "critic disabled"})

    # --- step 20: verification (mechanical re-run) ------------------- #
    if enabled["verification"]:
        verification = agents.verify_findings(db, findings)
        verified = [f for f in findings if f.verification_status == "verified"]
    else:
        # without verification every finding is simply taken at face value
        verification = [{"finding_id": str(f.id), "verified": None,
                         "reason": "verification disabled"} for f in findings]
        verified = list(findings)
    # Strongest explanation first: drivers before associations before plain
    # measurements, and within each group the more confident result first.
    _type_rank = {"driver": 0, "association": 1, "measurement": 2}
    _conf_rank = {"high": 0, "medium": 1, "low": 2}
    verified.sort(key=lambda f: (_type_rank.get(f.finding_type, 3),
                                 _conf_rank.get(f.confidence, 3),
                                 -abs(f.magnitude or 0)))

    # --- step 22: forecast ------------------------------------------- #
    forecasts: dict[str, Forecast] = {}
    # The observed series is kept alongside the projection so the report can
    # show where the forecast continues from. Without it a client can only draw
    # the dashed future, which is the least informative half of the picture.
    histories: dict[str, list] = {}
    if enabled["forecast"] and plan["date_column"] and verified:
        for finding in verified[:2]:
            result, _ = registry.call_tool(
                db, "run_forecast",
                {"dataset_id": str(investigation.dataset_id),
                 "version_id": str(investigation.version_id),
                 "date_column": plan["date_column"],
                 "metric": plan["target_metric"],
                 "horizon": forecast_horizon},
                investigation_id=investigation.id, agent_name="forecasting",
            )
            if "error" in result:
                continue
            fc = Forecast(
                investigation_id=investigation.id,
                finding_id=finding.id,
                target_metric=plan["target_metric"],
                model_name=result["model_name"],
                horizon_periods=result.get("horizon_periods"),
                frequency=result.get("frequency"),
                predictions={"points": result["predictions"]},
                backtest_metrics=result.get("backtest_metrics"),
                series_length=result.get("series_length"),
                reliability=result["reliability"],
                withheld_reason=result.get("withheld_reason"),
            )
            db.add(fc)
            db.flush()
            fc.predictions = result["predictions"]
            histories[str(fc.id)] = {
                "values": result.get("history") or [],
                "periods": result.get("history_periods") or [],
            }
            forecasts[str(finding.id)] = fc
            break
    db.flush()

    # --- step 23: recommendations ------------------------------------ #
    recs = (recommendations.generate_recommendations(
        db, investigation, verified, forecasts,
        require_verified=enabled["verification"])
        if enabled["recommendations"] else [])

    # --- step 23: report --------------------------------------------- #
    report = _build_report(db, investigation, verified, hypotheses, recs,
                           list(forecasts.values()), critique_result, history,
                           verification, plan, histories)

    # --- step 24: persist for future retrieval ----------------------- #
    rag.index_investigation_report(db, investigation, report)

    investigation.status = "awaiting_user" if requests else "complete"
    db.flush()
    return _state(db, investigation)


# --------------------------------------------------------------------- #
def resume(db: Session, investigation: Investigation, request_id: uuid.UUID,
           response: str) -> dict:
    """Proposal Sec.7 step 16 - validate the user's answer and continue."""
    req = db.get(MissingEvidenceRequest, request_id)
    if not req or req.investigation_id != investigation.id:
        raise ValueError("Request not found for this investigation")
    if not response.strip():
        raise ValueError("An empty response cannot be validated")

    req.response = response
    req.status = "answered"

    investigation.current_round += 1
    db.add(
        InvestigationRound(
            investigation_id=investigation.id,
            round_number=investigation.current_round,
            trigger="user_response",
            user_request=req.question,
            user_response=response,
            status="closed",
        )
    )

    # if the answer names a real column, adopt it and unblock the hypothesis
    version = db.get(DatasetVersion, investigation.version_id)
    df = ingestion.load_version(version)
    named = [c for c in df.columns if str(c).lower() in response.lower()]

    if named and req.request_type == "clarification" and not investigation.target_metric:
        investigation.target_metric = str(named[0])

    if req.hypothesis_id:
        h = db.get(Hypothesis, req.hypothesis_id)
        if h:
            if named:
                h.variables = {"columns": [str(c) for c in named]}
                h.status = "testable"
            else:
                h.status = "unresolved"
                h.reasoning = (
                    "The requested evidence was not available, so this hypothesis remains "
                    "unresolved rather than being treated as answered (proposal Sec.11)."
                )
    db.flush()
    return run(db, investigation)


def abandon(db: Session, investigation: Investigation, reason: str = "") -> dict:
    investigation.status = "abandoned"
    db.flush()
    return _state(db, investigation)


# --------------------------------------------------------------------- #
def _scoped_version(db: Session, investigation: Investigation,
                    version: DatasetVersion):
    """Materialise the investigation's date window as a derived version.

    Filtering only the orchestrator's in-memory dataframe would not work: every
    number comes from the tool layer, and the tools re-read the stored version
    from disk. So the window becomes a real derived version with its own
    lineage, and the investigation points at that instead.
    """
    scope = investigation.comparison_period or {}
    if not (scope.get("scope_start") or scope.get("scope_end")):
        return version, ingestion.load_version(version)

    # Always slice from the unscoped ancestor. Starting from another
    # investigation's window would compound two unrelated filters.
    base = version
    seen = set()
    while base.version_type == "scoped" and base.parent_version_id \
            and base.parent_version_id not in seen:
        seen.add(base.id)
        parent = db.get(DatasetVersion, base.parent_version_id)
        if not parent:
            break
        base = parent

    # reuse an identical window if this dataset already has one
    dataset_versions = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == investigation.dataset_id,
        DatasetVersion.version_type == "scoped").all()
    for candidate in dataset_versions:
        ops = candidate.cleaning_operations or {}
        if (ops.get("scope_start") == scope.get("scope_start")
                and ops.get("scope_end") == scope.get("scope_end")
                and ops.get("parent_version") == str(base.id)):
            investigation.version_id = candidate.id
            db.flush()
            return candidate, ingestion.load_version(candidate)

    version = base
    full = ingestion.load_version(version)
    scoped = _apply_period_scope(full, investigation)
    if len(scoped) == len(full):
        return version, full

    dataset = db.get(Dataset, investigation.dataset_id)
    derived = ingestion.new_version(
        db, dataset, scoped, version,
        version_type="scoped",
        cleaning_operations={
            "operation": "period_scope",
            "scope_start": scope.get("scope_start"),
            "scope_end": scope.get("scope_end"),
            "rows_before": len(full),
            "rows_after": len(scoped),
            "parent_version": str(version.id),
        },
        created_by_agent="supervisor",
        set_current=False,
    )
    investigation.version_id = derived.id
    db.flush()
    return derived, scoped


def _apply_period_scope(df, investigation: Investigation):
    """Restrict the dataframe to the investigation's date window, if it has one."""
    import pandas as pd

    from app.services.profiling import parse_datetimes

    scope = investigation.comparison_period or {}
    start, end = scope.get("scope_start"), scope.get("scope_end")
    if not start and not end:
        return df

    date_columns = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or any(h in str(c).lower() for h in ("date", "month", "time", "day"))
    ]
    if not date_columns:
        return df

    column = date_columns[0]
    parsed = parse_datetimes(df[column])
    mask = parsed.notna()
    if start:
        mask &= parsed >= pd.Timestamp(start)
    if end:
        mask &= parsed <= pd.Timestamp(end)

    scoped = df[mask]
    # a window that leaves almost nothing behind is a mistake, not a scope
    return scoped if len(scoped) >= 10 else df


def _build_report(db, investigation, findings, hypotheses, recs, forecasts,
                  critique_result, history, verification, plan=None,
                  histories=None) -> Report:
    plan = plan or {}
    histories = histories or {}
    existing = db.query(Report).filter(Report.investigation_id == investigation.id).count()
    summary = agents.write_summary(investigation, findings, critique_result)

    content = {
        "question": investigation.question,
        "target_metric": investigation.target_metric,
        "comparison_period": investigation.comparison_period,
        "findings": [
            {
                "id": str(f.id),
                "statement": f.statement,
                "finding_type": f.finding_type,
                "evidence_summary": f.evidence_summary,
                "magnitude": f.magnitude,
                "unit": f.unit,
                "confidence": f.confidence,
                "verification_status": f.verification_status,
                "caveats": f.caveats,
                "tool_run_id": str(f.tool_run_id) if f.tool_run_id else None,
            }
            for f in findings
        ],
        "hypotheses": [
            {"statement": h.statement, "status": h.status,
             "confidence": h.confidence, "reasoning": h.reasoning}
            for h in hypotheses
        ],
        "unresolved_hypotheses": [
            h.statement for h in hypotheses
            if h.status in {"unresolved", "blocked_missing_evidence"}
        ],
        "forecasts": [
            {
                "id": str(fc.id),
                "target_metric": fc.target_metric,
                "model": fc.model_name,
                "reliability": fc.reliability,
                "withheld_reason": fc.withheld_reason,
                "predictions": fc.predictions,
                "backtest": fc.backtest_metrics,
                "history": (histories.get(str(fc.id)) or {}).get("values", []),
                "history_periods": (histories.get(str(fc.id)) or {}).get("periods", []),
                "frequency": fc.frequency,
            }
            for fc in forecasts
        ],
        "recommendations": [
            {
                "rank": r.rank,
                "action": r.action,
                "rationale": r.rationale,
                "expected_impact": r.expected_impact,
                "impact_method": r.impact_method,
                "confidence": r.confidence,
                "urgency": r.urgency,
            }
            for r in recs
        ],
        "critique": critique_result,
        "historical_comparison": history,
        "retrieval": {
            "definitions_used": plan.get("definitions_used", []),
            "context_documents": plan.get("context_documents", []),
        },
        "verification": verification,
        "limitations": [
            "Correlation in this report is never presented as established causation.",
            "Forecasts assume current patterns continue; they are ranges, not guarantees.",
            "Recommendations are advisory. Each impact figure is a scenario "
            "estimate, not a measurement: it states what recovery would follow "
            "if the action closed the stated share of the gap. The arithmetic "
            "and the assumption behind it are shown with each recommendation.",
        ],
    }

    report = Report(
        investigation_id=investigation.id,
        version_number=existing + 1,
        title=f"Investigation report: {investigation.question[:150]}",
        executive_summary=summary,
        content=content,
        linked_finding_ids={"ids": [str(f.id) for f in findings]},
        status="final",
    )
    db.add(report)
    db.flush()
    return report


def _state(db: Session, investigation: Investigation) -> dict:
    open_requests = (
        db.query(MissingEvidenceRequest)
        .filter(MissingEvidenceRequest.investigation_id == investigation.id,
                MissingEvidenceRequest.status == "open")
        .all()
    )
    report = (
        db.query(Report)
        .filter(Report.investigation_id == investigation.id)
        .order_by(Report.version_number.desc())
        .first()
    )
    return {
        "investigation_id": str(investigation.id),
        "status": investigation.status,
        "round": investigation.current_round,
        "target_metric": investigation.target_metric,
        "open_requests": [
            {"id": str(r.id), "type": r.request_type, "question": r.question, "reason": r.reason}
            for r in open_requests
        ],
        "report": (
            {
                "id": str(report.id),
                "version": report.version_number,
                "executive_summary": report.executive_summary,
                **(report.content or {}),
            }
            if report else None
        ),
    }