"""Evaluation metrics (proposal Sec.20).

Every metric here is computed from a benchmark scenario whose true cause was
injected on purpose, so "correct" is a fact rather than a judgement call.

The two that matter most for the proposal's central claim:

  root_cause_rank_1      did the system name the true cause as its lead answer
  unsupported_claim_rate what share of stated conclusions have no tool run
                         behind them - this is where an LLM-only baseline is
                         expected to lose badly, and it is measurable rather
                         than rhetorical
"""

from __future__ import annotations


def _text_of(findings: list[dict]) -> str:
    return " ".join(
        f"{f.get('statement', '')} {f.get('evidence_summary', '')}" for f in findings
    ).lower()


def _lead(findings: list[dict]) -> dict | None:
    """The system's headline explanation: a driver or association, not a measurement."""
    causal = [f for f in findings
              if f.get("finding_type") in {"driver", "association"}]
    if causal:
        return causal[0]
    return findings[0] if findings else None


def _completed(run: dict) -> bool:
    """Did the run reach an answer at all?

    An error string or an offline placeholder is not an answer. This is checked
    before anything is scored, because the root-cause check works by searching
    the answer text for the true driver's name — and an error message can
    contain that name. A baseline whose reply was cut off mid-tool-call once
    returned "[error: Could not parse JSON ... SELECT product, AVG(in_stock_rate)
    ...]", which quoted the very column the scenario was about, and scored full
    marks for root cause while having completed nothing. A failed run earns no
    credit for a cause it never stated.
    """
    answer = run.get("answer") or ""
    return bool(answer) and not answer.startswith("[error") \
        and not answer.startswith("[mock")


def score_run(truth: dict, run: dict) -> dict:
    findings = run.get("findings") or []
    answer = (run.get("answer") or "").lower()
    completed = _completed(run)

    driver_col = (truth.get("true_driver_column") or "").lower()
    driver_val = (truth.get("true_driver_value") or "")
    driver_val = driver_val.lower() if driver_val else ""

    # ---------------- scenarios with no discoverable cause ---------------- #
    if truth.get("expected_behaviour", "").startswith("should raise"):
        asserted = any(
            f.get("finding_type") in {"driver", "association"}
            and f.get("confidence") == "high"
            for f in findings
        )
        asked = bool(run.get("asked_for_evidence")) or bool(
            run.get("unresolved_hypotheses"))
        # Crashing is not the same as declining to guess. Only a run that
        # finished can be credited with having asked instead of asserting.
        correct = completed and asked and not asserted
        return {
            "criterion": "asks instead of asserting",
            "asked_for_evidence": asked,
            "asserted_unsupported_cause": asserted,
            "root_cause_rank_1": correct,
            "root_cause_found": correct,
            **_shared(truth, run, findings, completed),
        }

    # ---------------- normal root-cause scenarios ------------------------- #
    # Only the run's own conclusions are searched. An incomplete run's text is
    # an error message, not a conclusion, so it is excluded from both checks.
    haystack = _text_of(findings) + (" " + answer if completed else "")
    lead = _lead(findings)
    lead_text = ((lead or {}).get("statement")
                 or (answer if completed else "")).lower()

    hit_lead = bool(driver_col and driver_col in lead_text) or bool(
        driver_val and driver_val in lead_text)
    hit_any = bool(driver_col and driver_col in haystack) or bool(
        driver_val and driver_val in haystack)

    return {
        "criterion": "true cause ranked first",
        "true_cause": truth["true_cause"],
        "lead_answer": (lead or {}).get("statement") or run.get("answer"),
        "root_cause_rank_1": completed and hit_lead,
        "root_cause_found": completed and hit_any,
        **_shared(truth, run, findings, completed),
    }


def forecast_accuracy(truth: dict, run: dict) -> dict:
    """Score forecasts against genuinely withheld actuals (proposal Sec.21).

    The model's own backtest MAPE measures fit on data it was allowed to see
    during selection. This measures the periods the generator removed from the
    file entirely, which is the harder and more honest test.
    """
    holdout = truth.get("forecast_holdout")
    forecasts = run.get("forecasts") or []
    if not holdout or not forecasts:
        return {"forecast_scored": False}

    predictions = forecasts[0].get("predictions") or []
    if not predictions:
        return {"forecast_scored": False,
                "forecast_withheld": forecasts[0].get("reliability") != "ok"}

    actuals = [p["actual"] for p in holdout["held_out_periods"]]
    points = [p["point"] for p in predictions][:len(actuals)]
    lowers = [p["lower"] for p in predictions][:len(actuals)]
    uppers = [p["upper"] for p in predictions][:len(actuals)]
    actuals = actuals[:len(points)]
    if not points:
        return {"forecast_scored": False}

    errors = [abs(a - p) / a * 100 for a, p in zip(actuals, points) if a]
    naive = actuals[0]
    naive_errors = [abs(a - naive) / a * 100 for a in actuals if a]
    covered = sum(low <= a <= high for a, low, high in zip(actuals, lowers, uppers))

    # Sec.20 asks for MAPE and RMSE; RMSE is in the metric's own units and
    # punishes a single large miss, which MAPE can hide.
    squared = [(a - p) ** 2 for a, p in zip(actuals, points)]
    naive_squared = [(a - naive) ** 2 for a in actuals]
    rmse = round((sum(squared) / len(squared)) ** 0.5, 2) if squared else None
    naive_rmse = round((sum(naive_squared) / len(naive_squared)) ** 0.5, 2) \
        if naive_squared else None

    holdout_mape = round(sum(errors) / len(errors), 2) if errors else None
    naive_mape = round(sum(naive_errors) / len(naive_errors), 2) if naive_errors else None
    return {
        "forecast_scored": True,
        "holdout_periods": len(points),
        "holdout_mape": holdout_mape,
        "holdout_naive_mape": naive_mape,
        "holdout_rmse": rmse,
        "holdout_naive_rmse": naive_rmse,
        "beats_naive_on_holdout": (holdout_mape is not None and naive_mape is not None
                                   and holdout_mape < naive_mape),
        # a range that never contains the truth is not a useful range
        "interval_coverage": round(covered / len(points), 3),
    }


def multi_source_credit(truth: dict, run: dict) -> dict:
    """Did the system use the companion document, not just the table?

    The multi_source scenario is designed so the table alone shows WHERE revenue
    fell but only the document says WHY, which is what separates a retrieval-
    capable architecture from one without.
    """
    if not truth.get("requires_document"):
        return {}
    terms = [t.lower() for t in truth.get("document_evidence_terms", [])]
    # an error message can quote a retrieved passage without the run having
    # used it, so an incomplete run's answer is left out here too
    answer = (run.get("answer") or "") if _completed(run) else ""
    haystack = (
        _text_of(run.get("findings") or []) + " " + answer
        + " " + " ".join(r.get("action", "") + " " + (r.get("rationale") or "")
                         for r in (run.get("recommendations") or []))
    ).lower()
    hits = [t for t in terms if t in haystack]
    retrieval = run.get("retrieval") or {}
    return {
        "document_evidence_terms_used": hits,
        "used_document_evidence": bool(hits),
        "documents_retrieved": len(retrieval.get("context_documents", [])),
    }


def driver_change_credit(truth: dict, run: dict) -> dict:
    """Did the system notice the driver is not the one it found last time?"""
    if not truth.get("tests_driver_change"):
        return {}
    comparison = ((run.get("historical_comparison") or {}).get("driver_comparison")
                  or {})
    changes = [c.get("driver_change") for c in comparison.get("comparisons", [])]
    return {
        "compared_with_previous": comparison.get("compared_with", 0),
        "reported_driver_change": "replaced" in changes,
    }


def _shared(truth: dict, run: dict, findings: list[dict],
            completed: bool | None = None) -> dict:
    """Metrics that apply to every scenario type."""
    if completed is None:
        completed = _completed(run)

    causal = [f for f in findings if f.get("finding_type") in {"driver", "association"}]
    verified = [f for f in findings if f.get("verification_status") == "verified"]
    traceable = [f for f in findings if f.get("tool_run_id")]

    # A conclusion with no tool run behind it cannot be checked. This is the
    # hallucination proxy: it does not ask whether the claim is true, only
    # whether anything in the system could tell.
    unsupported = [f for f in causal if not f.get("tool_run_id")]

    action = (truth.get("best_known_action") or "").lower()
    recommended = " ".join(
        r.get("action", "") for r in (run.get("recommendations") or [])).lower()
    action_words = [w for w in action.split() if len(w) > 4]
    action_hit = bool(action_words) and sum(
        w in recommended for w in action_words) >= max(1, len(action_words) // 3)

    forecasts = run.get("forecasts") or []
    return {
        "findings_total": len(findings),
        "causal_claims": len(causal),
        "verified_findings": len(verified),
        "traceable_findings": len(traceable),
        "traceability_rate": round(len(traceable) / len(findings), 3) if findings else 0.0,
        "unsupported_claim_rate": round(len(unsupported) / len(causal), 3) if causal else 0.0,
        "recommendation_matches_best_action": action_hit,
        "recommendations_made": len(run.get("recommendations") or []),
        "forecast_produced": bool(forecasts),
        "forecast_withheld_or_flagged": bool(
            forecasts and forecasts[0].get("reliability") != "ok"),
        "forecast_mape": (forecasts[0].get("backtest") or {}).get("mape")
        if forecasts else None,
        "tool_calls": run.get("tool_calls", 0),
        "llm_calls": run.get("llm_calls", 0),
        "prompt_chars": run.get("prompt_chars", 0),
        # ~4 characters per token is the usual rule of thumb; recorded as an
        # estimate because the real count depends on the provider's tokeniser
        "estimated_prompt_tokens": round(run.get("prompt_chars", 0) / 4),
        "latency_seconds": run.get("latency_seconds"),
        **forecast_accuracy(truth, run),
        **multi_source_credit(truth, run),
        **driver_change_credit(truth, run),
        "completed": completed,
    }


def aggregate(system: str, scored: list[dict]) -> dict:
    """Roll per-scenario scores up into one row of the comparison table."""
    n = len(scored) or 1

    def mean(key):
        values = [s[key] for s in scored if isinstance(s.get(key), (int, float))]
        return round(sum(values) / len(values), 3) if values else None

    return {
        "system": system,
        "scenarios": len(scored),
        "root_cause_accuracy": round(sum(bool(s["root_cause_rank_1"]) for s in scored) / n, 3),
        "root_cause_recall": round(sum(bool(s["root_cause_found"]) for s in scored) / n, 3),
        "traceability_rate": mean("traceability_rate"),
        "unsupported_claim_rate": mean("unsupported_claim_rate"),
        "recommendation_match_rate": round(
            sum(bool(s.get("recommendation_matches_best_action")) for s in scored) / n, 3),
        "task_completion_rate": round(sum(bool(s["completed"]) for s in scored) / n, 3),
        "avg_tool_calls": mean("tool_calls"),
        "avg_llm_calls": mean("llm_calls"),
        "avg_prompt_chars": mean("prompt_chars"),
        "avg_estimated_tokens": mean("estimated_prompt_tokens"),
        "avg_latency_seconds": mean("latency_seconds"),
        "holdout_mape": mean("holdout_mape"),
        "holdout_rmse": mean("holdout_rmse"),
        "forecast_interval_coverage": mean("interval_coverage"),
        "used_document_evidence": round(
            sum(bool(s.get("used_document_evidence")) for s in scored) / n, 3),
        "reported_driver_change": round(
            sum(bool(s.get("reported_driver_change")) for s in scored) / n, 3),
    }


COMPARISON_COLUMNS = [
    ("system", "System"),
    ("root_cause_accuracy", "Root cause @1"),
    ("root_cause_recall", "Found anywhere"),
    ("traceability_rate", "Traceable"),
    ("unsupported_claim_rate", "Unsupported"),
    ("recommendation_match_rate", "Rec match"),
    ("task_completion_rate", "Completed"),
    ("avg_tool_calls", "Tools"),
    ("avg_llm_calls", "LLM calls"),
    ("avg_latency_seconds", "Latency s"),
]


def format_table(rows: list[dict]) -> str:
    header = [label for _, label in COMPARISON_COLUMNS]
    widths = [
        max(len(label), max((len(str(r.get(key, ""))) for r in rows), default=0), 9)
        for key, label in COMPARISON_COLUMNS
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)),
             "  ".join("-" * w for w in widths)]
    for row in rows:
        cells = [str(row.get(key, "")) for key, _ in COMPARISON_COLUMNS]
        lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)