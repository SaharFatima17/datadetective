"""The investigation agents (proposal Sec.6, Sec.11, Sec.14, Sec.18).

Design note: the proposal lists sixteen agents, but Profiler, Cleaning, SQL and
Statistical are deterministic pipelines - they are implemented as TOOLS under
app/services and app/tools, not as LLM agents. Making them agents would add an
LLM hop, latency and a failure mode for no capability gain.

The agents that remain are the ones that genuinely need judgement:
  Hypothesis, EvidenceGap, Critic, History and Report - plus the Supervisor.
The Verifier is deliberately NOT an LLM: it re-executes tool runs mechanically.
"""

from __future__ import annotations

import uuid

import pandas as pd
from sqlalchemy.orm import Session

from app.llm.client import llm
from app.models import (
    Finding,
    Hypothesis,
    HypothesisEvidenceLink,
    Investigation,
    MetricSnapshot,
    MissingEvidenceRequest,
    ToolRun,
)
from app.services import rag
from app.tools import registry


# ===================================================================== #
# Hypothesis Agent (Sec.11)
# ===================================================================== #
HYPOTHESIS_SYSTEM = (
    "You generate testable hypotheses for a business data investigation. Each "
    "hypothesis must be checkable against tabular data using only the columns "
    "provided. Return a JSON array; each item has keys: statement, variables "
    "(array of column names that exist), proposed_test (string)."
)


def generate_hypotheses(db: Session, investigation: Investigation, plan: dict) -> list[Hypothesis]:
    available = plan["available"]
    all_cols = available["numeric"] + available["categorical"] + available["datetime"]

    try:
        raw = llm.complete_json(
            system=HYPOTHESIS_SYSTEM,
            prompt=(
                f"Generate hypotheses for this question: {investigation.question}\n\n"
                f"Target metric: {plan['target_metric']}\n"
                f"Date column: {plan['date_column']}\n"
                f"Numeric columns: {available['numeric']}\n"
                f"Categorical columns: {available['categorical']}\n\n"
                "Produce 4 to 6 distinct hypotheses covering different mechanisms."
            ),
        )
    except Exception:  # noqa: BLE001
        raw = []

    if not isinstance(raw, list):
        raw = []

    # Keep only hypotheses whose variables are real columns. A model that invents
    # column names would otherwise block the entire investigation behind
    # missing-evidence requests that are really its own hallucinations.
    usable = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("statement"):
            continue
        variables = [v for v in (item.get("variables") or []) if v in all_cols]
        if variables:
            usable.append({**item, "variables": variables})

    # Top up from the schema so there is always something testable to run.
    if len(usable) < 3:
        seen = {u["statement"] for u in usable}
        for item in _fallback_hypotheses(plan):
            if item["statement"] not in seen:
                usable.append(item)

    out = []
    for item in usable[:6]:
        h = Hypothesis(
            investigation_id=investigation.id,
            statement=str(item["statement"])[:2000],
            variables={"columns": item["variables"]},
            proposed_test=str(item.get("proposed_test", ""))[:1000],
            status="proposed",
        )
        db.add(h)
        out.append(h)
    db.flush()
    return out


def top_up_hypotheses(db: Session, investigation: Investigation, plan: dict,
                      existing: list[Hypothesis], available_columns: set[str]
                      ) -> list[Hypothesis]:
    """Generate hypotheses for columns that appeared after the first round.

    When a user answers a missing-evidence request by supplying data, the new
    column is useless unless something actually forms a hypothesis about it.
    Without this, a resumed investigation would merge the data and then report
    the same conclusion it had before.
    """
    covered = {
        c for h in existing for c in ((h.variables or {}).get("columns") or [])
    }
    new_columns = {c for c in available_columns if c not in covered}
    if not new_columns:
        return []

    metric = plan.get("target_metric")
    candidates = [
        item for item in _fallback_hypotheses(plan)
        if any(c in new_columns for c in item["variables"]) and metric
    ]
    if not candidates:
        return []

    seen = {h.statement for h in existing}
    created = []
    for item in candidates[:4]:
        if item["statement"] in seen:
            continue
        h = Hypothesis(
            investigation_id=investigation.id,
            statement=item["statement"],
            variables={"columns": item["variables"]},
            proposed_test=item.get("proposed_test", ""),
            status="proposed",
        )
        db.add(h)
        created.append(h)
    db.flush()
    return created


def _fallback_hypotheses(plan: dict) -> list[dict]:
    metric = plan["target_metric"]
    dims = plan["dimensions"]
    items = []
    for dim in dims[:3]:
        items.append(
            {
                "statement": f"The change in {metric} is concentrated in specific values of {dim}.",
                "variables": [dim, metric],
                "proposed_test": f"groupby_aggregate {metric} by {dim} and inspect contribution shares",
            }
        )
    for num in plan["available"]["numeric"]:
        if num != metric:
            items.append(
                {
                    "statement": f"{num} moves together with {metric} and may explain part of the change.",
                    "variables": [num, metric],
                    "proposed_test": f"correlation between {num} and {metric}",
                }
            )
            break
    if plan["date_column"]:
        items.append(
            {
                "statement": f"{metric} follows a time trend rather than a one-off shift.",
                "variables": [plan["date_column"], metric],
                "proposed_test": "time_series of the metric over the date column",
            }
        )
    return items


# ===================================================================== #
# Evidence Gap Agent (Sec.7 steps 14-16)
# ===================================================================== #
def check_evidence(
    db: Session, investigation: Investigation, hypotheses: list[Hypothesis], df: pd.DataFrame
) -> list[MissingEvidenceRequest]:
    """Marks hypotheses testable or blocked, and files a request for what is missing."""
    requests = []
    columns = set(str(c) for c in df.columns)

    for h in hypotheses:
        needed = set((h.variables or {}).get("columns") or [])
        missing = needed - columns

        if not needed:
            h.status = "blocked_missing_evidence"
            req = MissingEvidenceRequest(
                investigation_id=investigation.id,
                hypothesis_id=h.id,
                request_type="clarification",
                question=(
                    f"To test '{h.statement[:150]}' I need to know which column measures it. "
                    f"Available columns: {sorted(columns)}"
                ),
                reason="No dataset column could be matched to this hypothesis.",
            )
            db.add(req)
            requests.append(req)
        elif missing:
            h.status = "blocked_missing_evidence"
            req = MissingEvidenceRequest(
                investigation_id=investigation.id,
                hypothesis_id=h.id,
                request_type="missing_field",
                question=(
                    f"To test '{h.statement[:150]}' I need these fields which are not in "
                    f"the dataset: {sorted(missing)}. Can you supply them, or a dataset "
                    "that contains them?"
                ),
                reason=f"Missing columns: {sorted(missing)}",
            )
            db.add(req)
            requests.append(req)
        else:
            h.status = "testable"

    db.flush()
    return requests


# ===================================================================== #
# Analysis: runs the tools that test each hypothesis (Sec.7 step 17)
# ===================================================================== #
def test_hypothesis(
    db: Session, investigation: Investigation, h: Hypothesis, dataset_id: uuid.UUID,
    plan: dict, df: pd.DataFrame
) -> Finding | None:
    """Chooses the right tool for the hypothesis and records the result."""
    # Pin every tool call to the investigation's own version. Without this the
    # tools fall back to the dataset's current version, which may be a different
    # slice entirely - the bug that made two period-scoped investigations return
    # identical findings.
    version_id = str(investigation.version_id) if investigation.version_id else None

    cols = (h.variables or {}).get("columns") or []
    metric = plan["target_metric"]
    numeric = set(plan["available"]["numeric"])
    categorical = set(plan["available"]["categorical"])

    dim = next((c for c in cols if c in categorical), None)
    other_numeric = next((c for c in cols if c in numeric and c != metric), None)

    finding = None

    # --- segmentation hypothesis -------------------------------------- #
    # The question is almost never "which group is biggest" but "which group drove
    # the CHANGE". A group can be small overall and still cause the whole decline,
    # so when a date column exists this uses period contribution analysis
    # (proposal Sec.11, and the Sec.18 worked example) and falls back to plain
    # totals only when there is no usable time dimension.
    if dim and metric:
        if plan.get("date_column"):
            result, run = registry.call_tool(
                db, "run_dataframe_code",
                {"dataset_id": str(dataset_id), "operation": "period_contribution",
                 "params": {"date_column": plan["date_column"], "metric": metric, "by": dim},
                 "version_id": version_id},
                investigation_id=investigation.id, hypothesis_id=h.id, agent_name="analysis",
            )
        else:
            result, run = registry.call_tool(
                db, "run_dataframe_code",
                {"dataset_id": str(dataset_id), "operation": "groupby_aggregate",
                 "params": {"by": dim, "metric": metric, "agg": "sum"},
                 "version_id": version_id},
                investigation_id=investigation.id, hypothesis_id=h.id, agent_name="analysis",
            )

        if run.status == "success" and result.get("result"):
            if result["operation"] == "period_contribution":
                movers = [r for r in result["result"]
                          if r["contribution_to_total_change_pct"] > 0]
                top = movers[0] if movers else result["result"][0]
                share = top["contribution_to_total_change_pct"]
                n_groups = len(result["result"])
                even_share = 100 / max(n_groups, 1)
                direction = result["direction"]
                total_pct = abs(result.get("total_change_pct") or 0)

                # Three guards, all required:
                #  - the overall movement must be material, or "contribution" is noise
                #  - one group must carry clearly more than an even split
                #  - a share far above 100% means groups moved in opposite directions
                #    and cancelled out, so no single group "drove" anything
                material = total_pct >= 5
                concentrated = share >= even_share * 1.6
                coherent = share <= 150
                supported = material and concentrated and coherent

                if not supported:
                    if not material:
                        reject_reason = (
                            f"Overall {metric} moved only {total_pct:.1f}%, too small to "
                            "attribute to any group."
                        )
                    elif not coherent:
                        reject_reason = (
                            f"Groups moved in opposite directions and cancelled out "
                            f"(top share {share}%), so the change is not concentrated in one group."
                        )
                    else:
                        reject_reason = (
                            f"No group carries a disproportionate share "
                            f"(top {share}% against an even split of {even_share:.1f}%)."
                        )

                stat = None
                if supported and n_groups >= 2:
                    others = [r["group"] for r in result["result"] if r["group"] != top["group"]]
                    if others:
                        stat_result, stat_run = registry.call_tool(
                            db, "run_statistical_test",
                            {"dataset_id": str(dataset_id), "test": "ttest",
                             "params": {"value_column": metric, "group_column": dim,
                                        "groups": [top["group"], others[0]]},
                             "version_id": version_id},
                            investigation_id=investigation.id, hypothesis_id=h.id,
                            agent_name="analysis",
                        )
                        if stat_run.status == "success" and "error" not in stat_result:
                            stat = stat_result
                            _link(db, h, "tool_run", stat_run.id,
                                  "supports" if stat.get("significant") else "inconclusive")

                h.status = "supported" if supported else "rejected"
                h.confidence = round(min(share / 100, 0.95), 2) if supported else 0.2
                h.reasoning = (
                    f"{dim}='{top['group']}' accounts for {share}% of the total "
                    f"{direction} across {n_groups} groups "
                    f"({top['previous_per_period']} -> {top['current_per_period']} per period)."
                    if supported else reject_reason
                )
                _link(db, h, "tool_run", run.id, "supports" if supported else "rejects")

                if supported:
                    finding = Finding(
                        investigation_id=investigation.id,
                        hypothesis_id=h.id,
                        tool_run_id=run.id,
                        statement=(
                            f"{dim} = '{top['group']}' contributed {share}% of the total "
                            f"{metric} {direction} (from {top['previous_per_period']} to "
                            f"{top['current_per_period']} per period, "
                            f"{top['change_pct']}%)."
                        ),
                        evidence_summary=(
                            f"period_contribution({metric} by {dim}, split "
                            f"{result['split_date']}) over {len(df)} rows; values normalised "
                            f"per period. Overall {direction} "
                            f"{result['total_change_per_period']} "
                            f"({result['total_change_pct']}%)."
                            + (f" Welch t-test p={stat['p_value']:.4g}, "
                               f"Cohen's d={stat['cohens_d']}." if stat else "")
                            + _split_sentence(result)
                            + _season_sentence(result.get("seasonality"))
                            + _interval_sentence(top.get("contribution_ci"))
                        ),
                        finding_type="driver",
                        magnitude=abs(float(top["change"])),
                        unit=metric,
                        # A change that vanishes against the same months a year
                        # earlier is a calendar, not a cause. It is not thrown
                        # away — the measurement stands — but it must not be
                        # presented with the same confidence as one that holds.
                        confidence=_driver_confidence(share, stat,
                                                      result.get("seasonality")),
                        caveats=_driver_caveats(result.get("seasonality")),
                    )
                    db.add(finding)
                    db.flush()
                    _refine_driver(db, investigation, plan, metric, dim,
                                   result["split_date"], finding)

            else:  # groupby_aggregate fallback - no date column available
                top = result["result"][0]
                share = top.get("share_pct", 0)
                supported = share >= (100 / max(len(result["result"]), 1)) * 1.6
                h.status = "supported" if supported else "rejected"
                h.confidence = round(min(share / 100, 0.9), 2) if supported else 0.2
                h.reasoning = f"{top['group']} accounts for {share}% of total {metric}."
                _link(db, h, "tool_run", run.id, "supports" if supported else "rejects")

                if supported:
                    finding = Finding(
                        investigation_id=investigation.id,
                        hypothesis_id=h.id,
                        tool_run_id=run.id,
                        statement=f"{dim} = '{top['group']}' accounts for {share}% of total {metric}.",
                        finding_type="driver",
                        evidence_summary=(
                            f"groupby_aggregate({metric} by {dim}, sum) over {len(df)} rows."
                        ),
                        magnitude=float(top["value"]),
                        unit=metric,
                        confidence="medium",
                        caveats=(
                            "No date column was available, so this measures overall size, "
                            "not change over time."
                        ),
                    )
                    db.add(finding)

    # --- correlation hypothesis --------------------------------------- #
    elif other_numeric and metric:
        result, run = registry.call_tool(
            db, "run_statistical_test",
            {"dataset_id": str(dataset_id), "test": "correlation",
             "params": {"column_a": other_numeric, "column_b": metric},
             "version_id": version_id},
            investigation_id=investigation.id, hypothesis_id=h.id, agent_name="analysis",
        )
        if run.status == "success":
            r = result.get("r", 0)
            significant = result.get("significant", False) and abs(r) > 0.3
            h.status = "supported" if significant else "rejected"
            h.confidence = round(min(abs(r), 0.9), 2)
            h.reasoning = f"Pearson r={r:.3f}, p={result.get('p_value', 1):.4g}"
            _link(db, h, "tool_run", run.id, "supports" if significant else "rejects")

            if significant:
                finding = Finding(
                    investigation_id=investigation.id,
                    hypothesis_id=h.id,
                    tool_run_id=run.id,
                    statement=(
                        f"{other_numeric} and {metric} are correlated "
                        f"(r={r:.3f}, n={result.get('n')})."
                    ),
                    finding_type="association",
                    evidence_summary=(
                        f"Pearson correlation over {result.get('n')} paired observations; "
                        f"r-squared={result.get('r_squared', 0):.3f}, p={result.get('p_value', 1):.4g}."
                    ),
                    magnitude=abs(float(r)),
                    unit="correlation coefficient",
                    confidence="medium",
                    caveats="Correlation does not establish causation (proposal Sec.19).",
                )
                db.add(finding)

    # --- time trend hypothesis ---------------------------------------- #
    elif plan["date_column"] and metric:
        result, run = registry.call_tool(
            db, "run_dataframe_code",
            {"dataset_id": str(dataset_id), "operation": "time_series",
             "params": {"date_column": plan["date_column"], "metric": metric, "freq": "M"},
             "version_id": version_id},
            investigation_id=investigation.id, hypothesis_id=h.id, agent_name="analysis",
        )
        if run.status == "success" and result.get("result"):
            points = [p["value"] for p in result["result"] if p["value"] is not None]
            if len(points) >= 3:
                change = points[-1] - points[0]
                pct = (change / points[0] * 100) if points[0] else 0
                trending = abs(pct) > 10
                h.status = "supported" if trending else "rejected"
                h.confidence = 0.7 if trending else 0.2
                h.reasoning = f"{metric} moved {pct:.1f}% from first to last period."
                _link(db, h, "tool_run", run.id, "supports" if trending else "rejects")

                if trending:
                    finding = Finding(
                        investigation_id=investigation.id,
                        hypothesis_id=h.id,
                        tool_run_id=run.id,
                        statement=f"{metric} changed by {pct:.1f}% across the observed period.",
                        finding_type="measurement",
                        evidence_summary=(
                            f"Monthly time series over {len(points)} periods: "
                            f"{points[0]:.2f} -> {points[-1]:.2f}."
                        ),
                        magnitude=abs(float(change)),
                        unit=metric,
                        confidence="high",
                        caveats=(
                            "This measures WHAT changed, not why. It is not an explanation "
                            "and cannot support a recommendation on its own."
                        ),
                    )
                    db.add(finding)
    else:
        h.status = "unresolved"
        h.reasoning = "No suitable test could be selected for the variables in this hypothesis."

    db.flush()
    return finding


def _link(db: Session, h: Hypothesis, evidence_type: str, evidence_id, relation: str) -> None:
    db.add(
        HypothesisEvidenceLink(
            hypothesis_id=h.id, evidence_type=evidence_type,
            evidence_id=evidence_id, relation=relation,
        )
    )


# ===================================================================== #
# Critic Agent (Sec.7 step 19)
# ===================================================================== #
CRITIC_SYSTEM = (
    "You are a critical reviewer of data analysis. Challenge the conclusions: "
    "name confounders, alternative explanations, and claims the evidence does not "
    "support. Return JSON with keys: concerns (array of strings), verdict "
    "(one of: proceed, proceed_with_caveats, insufficient_evidence)."
)


def critique(db: Session, investigation: Investigation, findings: list[Finding]) -> dict:
    if not findings:
        return {"concerns": ["No findings were produced, so there is nothing to support a conclusion."],
                "verdict": "insufficient_evidence"}

    summary = "\n".join(f"- {f.statement} (evidence: {f.evidence_summary})" for f in findings)
    try:
        result = llm.complete_json(
            system=CRITIC_SYSTEM,
            prompt=f"Question: {investigation.question}\n\nFindings:\n{summary}\n\nCritique these.",
        )
    except Exception:  # noqa: BLE001
        result = {"concerns": ["Automated critique unavailable."], "verdict": "proceed_with_caveats"}

    concerns = result.get("concerns", [])
    for f in findings:
        if f.confidence == "high" and "correlation" in (f.statement or "").lower():
            concerns.append(
                f"'{f.statement[:80]}' is a correlation; its confidence should not be high."
            )
            f.confidence = "medium"
    result["concerns"] = concerns
    db.flush()
    return result


# ===================================================================== #
# Verifier - deliberately NOT an LLM (Sec.7 step 20)
# ===================================================================== #
def _refine_driver(db, investigation, plan: dict, metric: str, primary: str,
                   split: str, parent) -> None:
    """Narrow a driver to a pair of columns when the movement sits in one cell.

    "The decline is in the South" is true and often where an analysis stops.
    But the South may be four products of which one collapsed and three are
    healthy — and acting on "the South" then spends effort on the three. This
    looks one level deeper and records a second finding only when the movement
    really is concentrated there.
    """
    others = [d for d in (plan.get("dimensions") or []) if d != primary]
    if not others:
        return

    # "South, and within it product Y" and "product Y, and within it the South"
    # are the same cell said twice. Refine once per investigation — the point is
    # to narrow the leading answer, not to restate it from every angle.
    already = (
        db.query(Finding)
        .filter(Finding.investigation_id == investigation.id,
                Finding.evidence_summary.like("interaction_contribution%"))
        .first()
    )
    if already:
        return

    result, run = registry.call_tool(
        db, "run_dataframe_code",
        {"dataset_id": str(investigation.dataset_id),
         "version_id": str(investigation.version_id),
         "operation": "interaction_contribution",
         "params": {"date_column": plan["date_column"], "metric": metric,
                    "by": [primary, others[0]], "split": split}},
        investigation_id=investigation.id, agent_name="analysis")

    if run.status != "success" or not result.get("worth_reporting"):
        return

    inner = result["within_leading_group"][0]
    db.add(Finding(
        investigation_id=investigation.id,
        hypothesis_id=parent.hypothesis_id,
        tool_run_id=run.id,
        statement=(
            f"Within {primary} = '{result['leading_group']}', the "
            f"{result['direction']} is concentrated in {others[0]} = "
            f"'{inner['group']}' ({result['refined_share_within_group_pct']}% of "
            f"that group's movement)."
        ),
        finding_type="driver",
        evidence_summary=(
            f"interaction_contribution({metric} by {primary} then {others[0]}, "
            f"split {result['split_date']}) across {result['values_examined']} "
            f"values of {others[0]}."
        ),
        magnitude=abs(float(inner.get("change") or 0)),
        unit=metric,
        confidence=parent.confidence,
        verification_status="pending",
        caveats=(
            "This narrows where the movement sits; it still does not say why. "
            "It is reported only because one value carries most of the change — "
            "had it been spread evenly, the broader statement would stand alone."
        ),
    ))


def _split_sentence(result: dict) -> str:
    """Say how the comparison period was chosen.

    A contribution figure depends entirely on where the line between "before"
    and "after" is drawn. Reporting the number without saying how that date was
    picked invites the reader to assume it was found in the data when, before
    change-point detection, it was simply the most recent third.
    """
    method = result.get("split_method")
    date = result.get("split_date")
    if method == "detected":
        return (f" The split at {date} is where the series changes level, found "
                f"by scanning every candidate date (explains "
                f"{result.get('split_strength')} of the variance).")
    if method == "fallback":
        return (f" No clear change point was found, so the split at {date} is "
                "the most recent third of the range rather than a detected "
                "event. Treat the date as arbitrary.")
    if method == "supplied":
        return f" The split at {date} was supplied rather than detected."
    return ""


def _season_sentence(seasonality: dict | None) -> str:
    if not seasonality or not seasonality.get("checked"):
        return ""
    yoy = seasonality.get("year_on_year_pct")
    if yoy is None:
        return ""
    verdict = ("which is not explained by the season alone"
               if seasonality.get("survives")
               else "which is roughly flat, so the season accounts for most of it")
    return (f" Against the same {seasonality['months_compared']} calendar months "
            f"a year earlier the change is {yoy}%, {verdict}.")


def _interval_sentence(ci: dict | None) -> str:
    if not ci:
        return ""
    return (f" The share is an estimate: 95% interval "
            f"{ci['low']}% to {ci['high']}% ({ci['method']}).")


def _driver_confidence(share: float, stat: dict | None,
                       seasonality: dict | None) -> str:
    if seasonality and seasonality.get("checked") and not seasonality.get("survives"):
        return "low"
    if share > 50 and stat and stat.get("significant"):
        return "high"
    return "medium"


def _driver_caveats(seasonality: dict | None) -> str:
    base = "Contribution shows where the change is concentrated, not why. "
    if not seasonality or not seasonality.get("checked"):
        reason = (seasonality or {}).get("reason", "not enough history")
        return (base + "Seasonality could not be ruled out: " + reason + ". "
                "Period length is normalised, the season is not.")
    if seasonality.get("survives"):
        return (base + "The change does survive a like-for-like comparison with "
                "the same months a year earlier, so it is not the season alone.")
    return (base + "Against the same months a year earlier this change is close "
            "to flat, so most of it is seasonal. Treat this as a pattern that "
            "repeats, not a new cause.")


def verify_findings(db: Session, findings: list[Finding]) -> list[dict]:
    """Re-executes each finding's recorded tool run and compares checksums."""
    results = []
    for f in findings:
        if not f.tool_run_id:
            f.verification_status = "failed_verification"
            results.append({"finding_id": str(f.id), "verified": False,
                            "reason": "no tool run linked to this finding"})
            continue
        run = db.get(ToolRun, f.tool_run_id)
        outcome = registry.reverify(db, run)
        f.verification_status = "verified" if outcome["verified"] else "failed_verification"
        results.append({"finding_id": str(f.id), "statement": f.statement, **outcome})
    db.flush()
    return results


# ===================================================================== #
# History / Comparison Agent (Sec.14, Sec.7 steps 18 & 21)
# ===================================================================== #
def compare_with_history(db: Session, investigation: Investigation, findings: list[Finding]) -> dict:
    prior = rag.search(db, investigation.question, top_k=3,
                       document_type="past_report", owner_id=investigation.owner_id)
    prior = [p for p in prior
             if (p.get("document_title") or "") and str(investigation.id) not in str(p)]

    if not prior:
        return {"has_history": False,
                "note": "No previous investigation on this question was found in the knowledge base."}

    return {
        "has_history": True,
        "previous": prior,
        "current_findings": [f.statement for f in findings],
        "note": (
            "Previous findings are retrieved as context and must be validated against "
            "current evidence, not presented as current truth (proposal Sec.14)."
        ),
    }


def snapshot_metrics(db: Session, investigation: Investigation, plan: dict,
                     df: pd.DataFrame) -> list[MetricSnapshot]:
    """Records current KPI values so a future investigation can compare (Sec.15)."""
    metric = plan["target_metric"]
    if not metric or metric not in df.columns:
        return []

    snaps = [
        MetricSnapshot(
            investigation_id=investigation.id,
            dataset_version_id=investigation.version_id,
            metric_name=metric,
            value=float(df[metric].sum()),
            calculation_definition=f"sum({metric}) over {len(df)} rows",
        )
    ]
    for dim in plan["dimensions"][:2]:
        if dim in df.columns:
            for group, value in df.groupby(dim)[metric].sum().items():
                snaps.append(
                    MetricSnapshot(
                        investigation_id=investigation.id,
                        dataset_version_id=investigation.version_id,
                        metric_name=metric,
                        segment={dim: str(group)},
                        value=float(value),
                        calculation_definition=f"sum({metric}) where {dim}='{group}'",
                    )
                )
    for s in snaps:
        db.add(s)
    db.flush()
    return snaps


# ===================================================================== #
# Report Agent (Sec.18)
# ===================================================================== #
REPORT_SYSTEM = (
    "You write the executive summary of a data investigation report. Use ONLY the "
    "findings given. Never introduce a number that is not in them. Separate what was "
    "measured from what is interpretation. Plain prose, 3-5 sentences."
)


def write_summary(investigation: Investigation, findings: list[Finding],
                  critique_result: dict) -> str:
    if not findings:
        return (
            f"No verified findings were produced for the question '{investigation.question}'. "
            "The hypotheses that were generated could not be tested against the available data, "
            "so no conclusion is offered. The unresolved hypotheses and the evidence each one "
            "requires are listed below."
        )
    body = "\n".join(f"- {f.statement} [{f.confidence} confidence]" for f in findings)
    try:
        return llm.complete(
            system=REPORT_SYSTEM,
            prompt=(
                f"Question: {investigation.question}\n\nVerified findings:\n{body}\n\n"
                f"Reviewer concerns: {critique_result.get('concerns', [])}\n\n"
                "Write the executive summary."
            ),
        ).strip()
    except Exception:  # noqa: BLE001
        return (
            f"Investigation of '{investigation.question}' produced {len(findings)} verified "
            f"finding(s). Each is computed directly from the dataset and linked to the "
            f"calculation that produced it. Reviewer concerns are listed separately and "
            f"should be read alongside the findings."
        )