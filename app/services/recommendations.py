"""Phase 10 - Recommendations (proposal Sec.13).

The impact number is computed here, deterministically, and the exact formula plus
its assumptions are stored alongside it in `impact_method`. The LLM only writes
the sentence around the number.

This addresses the weakest claim in the proposal: "recovering X% of lost revenue"
is a counterfactual, not a measurement. So it is reported as an assumption-driven
projection with the arithmetic shown, never as a causal estimate.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.llm.client import llm
from app.models import Finding, Forecast, Investigation, Recommendation
from app.services import rag

# how much of an identified gap a corrective action is assumed to close
RECOVERY_ASSUMPTIONS = {
    "high": (0.55, 0.80),
    "medium": (0.35, 0.60),
    "low": (0.15, 0.35),
}


def estimate_impact(finding: Finding, forecast: Forecast | None) -> dict:
    """Deterministic impact arithmetic with the assumption stated explicitly."""
    magnitude = float(finding.magnitude or 0.0)
    confidence = (finding.confidence or "medium").lower()
    low_frac, high_frac = RECOVERY_ASSUMPTIONS.get(confidence, RECOVERY_ASSUMPTIONS["medium"])

    projected_further_loss = 0.0
    if forecast and forecast.predictions and forecast.reliability == "ok":
        points = [p["point"] for p in forecast.predictions]
        if points:
            baseline = points[0]
            projected_further_loss = float(max(0.0, baseline - min(points)))

    addressable = abs(magnitude) + projected_further_loss

    return {
        "low": round(addressable * low_frac, 2),
        "high": round(addressable * high_frac, 2),
        "unit": finding.unit or "units of the target metric",
        "addressable_gap": round(addressable, 2),
        "recovery_fraction_assumed": [low_frac, high_frac],
    }


def impact_method_text(finding: Finding, forecast: Forecast | None, impact: dict) -> str:
    parts = [
        "Impact range = addressable gap x assumed recovery fraction.",
        f"Addressable gap = measured effect ({abs(float(finding.magnitude or 0)):.2f})",
    ]
    if forecast and forecast.reliability == "ok":
        parts.append(f"+ projected further movement from forecast {forecast.id}")
    parts.append(
        f"Recovery fraction {impact['recovery_fraction_assumed'][0]:.0%}-"
        f"{impact['recovery_fraction_assumed'][1]:.0%} is an ASSUMPTION tied to the "
        f"finding's confidence level ({finding.confidence}), not a measured or causal quantity."
    )
    parts.append(
        "This is a scenario estimate: it states what recovery would follow IF the "
        "action closes that share of the gap. It does not establish that the action "
        "will cause the recovery."
    )
    return " ".join(parts)


def rank_score(impact: dict, confidence: str, urgency: str) -> float:
    conf_w = {"high": 1.0, "medium": 0.65, "low": 0.35}.get((confidence or "medium").lower(), 0.65)
    urg_w = {"high": 1.0, "medium": 0.7, "low": 0.4}.get((urgency or "medium").lower(), 0.7)
    midpoint = (impact["low"] + impact["high"]) / 2
    return midpoint * conf_w * urg_w


def generate_recommendations(
    db: Session, investigation: Investigation, findings: list[Finding],
    forecasts: dict[str, Forecast] | None = None,
    require_verified: bool = True,
) -> list[Recommendation]:
    """Only verified findings produce recommendations (proposal Sec.13).

    `require_verified=False` exists solely for the Sec.20 verification ablation.
    Without it, disabling the verifier would also silently disable
    recommendations, and the experiment would be measuring two changes at once.
    It is never relaxed in normal operation.
    """
    forecasts = forecasts or {}
    # Only DRIVER findings can produce an action (proposal Sec.13: "unverified
    # hypotheses never produce a recommendation"). A measurement says what changed
    # and an association says two things move together - neither identifies
    # something to act on, and "recover 0.4 of a correlation" is meaningless.
    verified = [
        f for f in findings
        if (f.verification_status == "verified" or not require_verified)
        and f.finding_type == "driver"
        and (f.unit or "").lower() not in {"correlation coefficient", ""}
    ]
    out: list[Recommendation] = []

    for finding in verified:
        forecast = forecasts.get(str(finding.id))
        impact = estimate_impact(finding, forecast)

        # Historical grounding (proposal Sec.13 point 4): past reports for
        # context, and past FEEDBACK on similar drivers for whether the action
        # actually helped last time.
        prior = rag.search(db, finding.statement, top_k=2, document_type="past_report")
        history = rag.past_feedback_for_driver(db, finding.statement)

        notes = []
        if prior:
            notes.append(f"Similar prior investigation: {prior[0]['document_title']}")
        if history["matched"]:
            notes.append(
                f"A similar action was rated useful in {history['useful']} of "
                f"{history['matched']} past investigation(s)."
            )
        prior_note = " ".join(notes) if notes else None

        urgency = "medium"
        if forecast and forecast.reliability == "ok" and forecast.predictions:
            points = [p["point"] for p in forecast.predictions]
            if len(points) > 1 and points[-1] < points[0]:
                urgency = "high"

        try:
            drafted = llm.complete_json(
                system=(
                    "You draft business recommendations. You are given a VERIFIED finding "
                    "and a computed impact range. Never invent or alter numbers - use only "
                    "what is given. Return JSON with keys: action, rationale, urgency."
                ),
                prompt=(
                    f"Verified finding: {finding.statement}\n"
                    f"Evidence: {finding.evidence_summary}\n"
                    f"Computed impact range: {impact['low']} to {impact['high']} {impact['unit']}\n"
                    f"Forecast trend: {'declining' if urgency == 'high' else 'stable'}\n"
                    f"{prior_note or ''}\n\n"
                    "Write one specific recommendation."
                ),
            )
        except Exception:  # noqa: BLE001
            drafted = {
                "action": f"Address the driver identified in: {finding.statement}",
                "rationale": finding.evidence_summary or "",
                "urgency": urgency,
            }

        rec = Recommendation(
            investigation_id=investigation.id,
            finding_id=finding.id,
            forecast_id=forecast.id if forecast else None,
            action=str(drafted.get("action", ""))[:2000],
            rationale=str(drafted.get("rationale", "")),
            expected_impact=impact,
            impact_method=impact_method_text(finding, forecast, impact),
            confidence=finding.confidence,
            urgency=str(drafted.get("urgency", urgency)),
        )
        if history["matched"]:
            rec.rationale = (
                (rec.rationale or "")
                + f" A similar action was rated useful in {history['useful']} of "
                  f"{history['matched']} past investigation(s)."
            ).strip()
        db.add(rec)
        out.append(rec)

    db.flush()
    ranked = sorted(
        out, key=lambda r: rank_score(r.expected_impact, r.confidence, r.urgency), reverse=True
    )
    for i, rec in enumerate(ranked, 1):
        rec.rank = i
    db.flush()
    return ranked
