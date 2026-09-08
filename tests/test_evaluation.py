"""Tests for the Sec.20 evaluation layer and the Sec.9 tool registry.

The metrics tests use hand-built run dictionaries rather than live runs, so they
assert on the scoring logic itself - the part that decides whether the proposal's
central claim is supported.
"""

from __future__ import annotations

import pytest

from app.evaluation import metrics
from app.tools import registry

TRUTH = {
    "scenario": "regional_decline",
    "true_cause": "Revenue decline is concentrated in region South",
    "true_driver_column": "region",
    "true_driver_value": "South",
    "target_metric": "revenue",
    "best_known_action": "Investigate and restore performance in the South region",
}

TRUTH_NO_CAUSE = {
    "scenario": "missing_evidence",
    "true_cause": "Not determinable from this dataset",
    "true_driver_column": None,
    "true_driver_value": None,
    "expected_behaviour": "should raise a missing-evidence request, not assert a cause",
    "best_known_action": "Request additional operational data before concluding",
}


# ------------------------------------------------------------- tool layer
def test_every_proposal_tool_is_registered():
    """Proposal Sec.9 lists these; each must exist and be described."""
    required = {
        "inspect_dataset", "profile_columns", "run_dataframe_code",
        "run_readonly_sql", "apply_cleaning_plan", "vector_search",
        "get_business_definition", "save_finding", "render_chart", "run_forecast",
        "generate_recommendations", "save_recommendation", "ingest_source",
        "extract_document", "fetch_url", "request_missing_evidence",
        "resume_investigation", "compare_investigations", "save_report",
    }
    missing = required - set(registry.TOOLS)
    assert not missing, f"tools from proposal Sec.9 not registered: {sorted(missing)}"
    assert not set(registry.TOOLS) - set(registry.TOOL_DESCRIPTIONS)


def test_mcp_exposes_exactly_the_registry():
    """The MCP surface and the internal registry must not drift apart."""
    import mcp_server

    exposed = {fn.__name__ for fn in mcp_server.EXPOSED}
    assert exposed == set(registry.TOOLS)


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.call_tool(None, "delete_everything", {})


# ---------------------------------------------------------------- metrics
def test_lead_answer_ignores_a_plain_measurement():
    """'Revenue fell 34%' describes the change; it is not an explanation."""
    findings = [
        {"statement": "revenue changed by -34%", "finding_type": "measurement"},
        {"statement": "region = 'South' contributed 99% of the decline",
         "finding_type": "driver"},
    ]
    assert metrics._lead(findings)["finding_type"] == "driver"


def test_correct_driver_scores_rank_1():
    run = {
        "answer": "",
        "findings": [{"statement": "region = 'South' contributed 99.1% of the decline",
                      "finding_type": "driver", "confidence": "high",
                      "verification_status": "verified", "tool_run_id": "abc"}],
        "tool_calls": 5,
    }
    scored = metrics.score_run(TRUTH, run)
    assert scored["root_cause_rank_1"] is True
    assert scored["root_cause_found"] is True
    assert scored["unsupported_claim_rate"] == 0.0
    assert scored["traceability_rate"] == 1.0


def test_wrong_driver_does_not_score():
    run = {"answer": "", "findings": [
        {"statement": "product = 'Alpha' contributed 40%", "finding_type": "driver",
         "verification_status": "verified", "tool_run_id": "abc"}]}
    assert metrics.score_run(TRUTH, run)["root_cause_rank_1"] is False


def test_untraceable_claim_counts_as_unsupported():
    """This is the metric that separates an LLM-only baseline from the rest."""
    run = {"answer": "The South region caused it",
           "findings": [{"statement": "The South region caused it",
                         "finding_type": "driver", "confidence": "high",
                         "verification_status": "failed_verification",
                         "tool_run_id": None}]}
    scored = metrics.score_run(TRUTH, run)
    assert scored["root_cause_rank_1"] is True      # it happens to be right
    assert scored["unsupported_claim_rate"] == 1.0  # but nothing backs it up
    assert scored["traceability_rate"] == 0.0


def test_asking_beats_asserting_when_no_cause_exists():
    asking = {"answer": "I need more data", "findings": [
        {"statement": "revenue changed by -37%", "finding_type": "measurement",
         "confidence": "high", "tool_run_id": "abc"}],
        "asked_for_evidence": True}
    asserting = {"answer": "It was the economy", "findings": [
        {"statement": "the economy caused it", "finding_type": "driver",
         "confidence": "high", "tool_run_id": None}],
        "asked_for_evidence": False}

    assert metrics.score_run(TRUTH_NO_CAUSE, asking)["root_cause_rank_1"] is True
    assert metrics.score_run(TRUTH_NO_CAUSE, asserting)["root_cause_rank_1"] is False


def test_a_measurement_alone_is_not_an_asserted_cause():
    """Stating what changed is correct behaviour, not a hallucination."""
    run = {"answer": "revenue fell", "findings": [
        {"statement": "revenue changed by -37%", "finding_type": "measurement",
         "confidence": "high", "tool_run_id": "abc"}],
        "asked_for_evidence": True}
    assert metrics.score_run(TRUTH_NO_CAUSE, run)["asserted_unsupported_cause"] is False


def test_error_answers_do_not_count_as_completed():
    for answer in ("[error: boom]", "[mock provider: nothing]"):
        scored = metrics.score_run(TRUTH, {"answer": answer, "findings": []})
        assert scored["completed"] is False


def test_aggregate_builds_a_comparison_row():
    scored = [
        metrics.score_run(TRUTH, {"answer": "x", "findings": [
            {"statement": "region = 'South' drove it", "finding_type": "driver",
             "verification_status": "verified", "tool_run_id": "a"}]}),
        metrics.score_run(TRUTH, {"answer": "y", "findings": [
            {"statement": "product = 'Alpha' drove it", "finding_type": "driver",
             "verification_status": "verified", "tool_run_id": "b"}]}),
    ]
    row = metrics.aggregate("proposed", scored)
    assert row["scenarios"] == 2
    assert row["root_cause_accuracy"] == 0.5
    assert row["task_completion_rate"] == 1.0
    assert "System" in metrics.format_table([row])


# -------------------------------------------------------------- ablations
def test_all_ablation_switches_are_known():
    from app.agents.orchestrator import ABLATIONS
    from scripts.run_comparison import ABLATION_VARIANTS

    for variant in ABLATION_VARIANTS.values():
        assert set(variant) <= set(ABLATIONS)


def test_baseline_registry_covers_the_four_architectures():
    from app.evaluation import baselines

    assert set(baselines.SYSTEMS) == {"baseline_a", "baseline_b", "baseline_c", "proposed"}


def test_baseline_c_has_retrieval_that_baseline_b_lacks():
    """The only difference between B and C must be retrieval (proposal Sec.20)."""
    from app.evaluation import baselines

    extra = set(baselines.C_TOOLS) - set(baselines.B_TOOLS)
    assert extra == {"vector_search", "get_business_definition"}
