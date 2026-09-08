"""The remaining proposal Sec.20 metrics.

These are kept apart from `metrics.py` because they score different things: not
"did the system reach the right conclusion", but "did the layers underneath it
behave correctly" - profiling, cleaning, hypothesis handling, test selection and
numerical accuracy.

Everything here is scored against defects or values injected on purpose by
`scripts/generate_benchmark.py`, so correctness is a fact rather than a
judgement.
"""

from __future__ import annotations

import math

import pandas as pd

# A detected issue name may legitimately differ from the injected label; these
# are the ones that mean the same defect.
EQUIVALENT = {
    "outliers": {"outliers", "extreme_outliers"},
    "anomalous_rows": {"anomalous_rows"},
    "date_problems": {"date_problems", "implausible_dates", "timezone_inconsistency"},
    "duplicate_rows": {"duplicate_rows"},
    "duplicate_keys": {"duplicate_keys"},
    "missing_values": {"missing_values"},
    "invalid_range": {"invalid_range"},
    "inconsistent_categories": {"inconsistent_categories"},
    "similar_categories": {"similar_categories"},
    "inconsistent_units": {"inconsistent_units"},
}


def detected_pairs(profile: dict) -> set[tuple[str, str]]:
    """Flatten a profile into (column, issue) pairs, dataset-level issues included."""
    pairs: set[tuple[str, str]] = set()
    for issue in profile.get("dataset_issues", []):
        column = issue.get("column") or "__dataset__"
        pairs.add((column, issue["issue"]))
    for column in profile.get("columns", []):
        for issue in column.get("issues", []):
            pairs.add((column["name"], issue["issue"]))
    return pairs


def _matches(expected: tuple[str, str], found: set[tuple[str, str]]) -> bool:
    column, issue = expected
    aliases = EQUIVALENT.get(issue, {issue})
    return any((column, alias) in found for alias in aliases)


def quality_detection(truth: dict, profile: dict) -> dict:
    """Precision and recall of data-quality detection (proposal Sec.20).

    Recall is the metric that matters: a missed defect corrupts every number
    computed afterwards. Precision is reported too, but a profiler that flags a
    few extra things costs an analyst a glance, not a wrong answer - so the two
    are not weighted equally in interpretation.
    """
    expected = [tuple(e) for e in truth.get("expected_issues", [])]
    if not expected:
        return {"quality_scored": False}

    found = detected_pairs(profile)
    hits = [e for e in expected if _matches(e, found)]
    missed = [e for e in expected if not _matches(e, found)]

    # anything detected that no expected defect explains
    explained: set[tuple[str, str]] = set()
    for column, issue in expected:
        for alias in EQUIVALENT.get(issue, {issue}):
            explained.add((column, alias))
    # constant/quality notes that are observations rather than defects
    # observations rather than defects: worth surfacing, but not something the
    # benchmark injected, so they must not count against precision
    benign = {"constant_column", "date_ordering", "anomalous_rows"}
    spurious = [p for p in found if p not in explained and p[1] not in benign]

    recall = len(hits) / len(expected)
    precision = len(hits) / max(len(hits) + len(spurious), 1)
    return {
        "quality_scored": True,
        "quality_expected": len(expected),
        "quality_detected": len(hits),
        "quality_recall": round(recall, 3),
        "quality_precision": round(precision, 3),
        "quality_f1": round(2 * precision * recall / (precision + recall), 3)
        if (precision + recall) else 0.0,
        "quality_missed": ["/".join(m) for m in missed],
        "quality_spurious": ["/".join(s) for s in spurious],
    }


def cleaning_correctness(truth: dict, before: pd.DataFrame, after: pd.DataFrame,
                         after_profile: dict) -> dict:
    """Did cleaning repair the defects without destroying valid data (Sec.20)?

    Two halves, and the second is the one that is easy to get wrong: a cleaner
    that drops every row scores perfectly on defect removal and is useless.
    """
    expected = [tuple(e) for e in truth.get("expected_issues", [])]
    if not expected:
        return {"cleaning_scored": False}

    by_design = {tuple(e) for e in truth.get("not_auto_repairable", [])}
    repairable = [e for e in expected if e not in by_design]

    remaining = detected_pairs(after_profile)
    resolved = [e for e in repairable if not _matches(e, remaining)]
    still_present_by_design = [e for e in by_design if _matches(e, remaining)]

    clean_rows = truth.get("clean_row_count")
    injected_dupes = truth.get("injected_duplicate_rows", 0)
    expected_rows = clean_rows if clean_rows else len(before) - injected_dupes

    retained = len(after) / expected_rows if expected_rows else 0.0
    # dropping rows that should have survived is over-cleaning; keeping the
    # injected duplicates is under-cleaning
    over_cleaned = max(0, expected_rows - len(after))
    under_cleaned = max(0, len(after) - expected_rows)

    return {
        "cleaning_scored": True,
        "defects_resolved": len(resolved),
        "defects_repairable": len(repairable),
        "defects_expected": len(expected),
        "defect_resolution_rate": round(len(resolved) / len(repairable), 3)
        if repairable else None,
        "left_unrepaired_by_design": ["/".join(e) for e in still_present_by_design],
        "rows_before": len(before),
        "rows_after": len(after),
        "expected_rows_after": expected_rows,
        "valid_data_retention": round(min(retained, 1.0), 3),
        "rows_over_cleaned": over_cleaned,
        "rows_under_cleaned": under_cleaned,
        "cleaning_correct": len(resolved) == len(repairable) and over_cleaned == 0,
    }


def numerical_accuracy(truth: dict, run: dict, tolerance: float = 0.05) -> dict:
    """Compare the reported magnitude against the injected one (proposal Sec.20).

    The system never estimates a number with an LLM, so this should be exact
    within rounding. A miss here means a tool is computing the wrong thing,
    which no amount of good reasoning would recover from.
    """
    expected = truth.get("true_magnitude")
    if expected is None:
        return {"numerical_scored": False}

    drivers = [f for f in (run.get("findings") or [])
               if f.get("finding_type") == "driver" and f.get("magnitude") is not None]
    if not drivers:
        return {"numerical_scored": True, "numerical_correct": False,
                "numerical_reported": None, "numerical_expected": expected}

    reported = float(drivers[0]["magnitude"])
    error = abs(reported - expected) / abs(expected) if expected else math.inf
    return {
        "numerical_scored": True,
        "numerical_expected": round(float(expected), 2),
        "numerical_reported": round(reported, 2),
        "numerical_relative_error": round(error, 4),
        "numerical_correct": error <= tolerance,
    }


def hypothesis_quality(run: dict) -> dict:
    """Hypothesis relevance and verification accuracy (proposal Sec.20).

    Relevance = the share of generated hypotheses that could actually be tested
    against the data. A system that emits ten untestable hypotheses has done
    nothing useful, even if one of them happens to be right.

    Verification accuracy = the share of findings whose claimed verification
    status was confirmed by re-running the recorded tool call.
    """
    hypotheses = run.get("hypotheses") or []
    verification = run.get("verification") or []

    if hypotheses:
        testable = [h for h in hypotheses
                    if h.get("status") in {"testable", "supported", "rejected"}]
        resolved = [h for h in hypotheses
                    if h.get("status") in {"supported", "rejected"}]
        relevance = len(testable) / len(hypotheses)
        resolution = len(resolved) / len(hypotheses)
    else:
        relevance = resolution = 0.0

    checked = [v for v in verification if v.get("verified") is not None]
    accuracy = (sum(bool(v["verified"]) for v in checked) / len(checked)
                if checked else None)

    return {
        "hypotheses_generated": len(hypotheses),
        "hypothesis_relevance": round(relevance, 3),
        "hypothesis_resolution_rate": round(resolution, 3),
        "verification_checked": len(checked),
        "verification_accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


# Which column types each test is valid for. Checked against what the tool was
# actually given, not against what the agent said it was doing.
TEST_REQUIREMENTS = {
    "ttest": {"value_column": "numeric", "group_column": "categorical"},
    "correlation": {"column_a": "numeric", "column_b": "numeric"},
    "chi_square": {"column_a": "categorical", "column_b": "categorical"},
    "linear_regression": {"y_column": "numeric"},
}


def test_appropriateness(run: dict, column_types: dict[str, str]) -> dict:
    """Was each statistical test valid for the columns it was run on (Sec.20)?

    A t-test on two categorical columns returns a number, and the number is
    meaningless. Nothing downstream would catch that, so it is checked here.
    """
    calls = [c for c in (run.get("tool_runs") or [])
             if c.get("tool_name") == "run_statistical_test"]
    if not calls:
        return {"tests_run": 0, "test_appropriateness": None}

    appropriate, problems = 0, []
    for call in calls:
        params = (call.get("parameters") or {})
        test = params.get("test")
        inner = params.get("params") or {}
        requirements = TEST_REQUIREMENTS.get(test)
        if not requirements:
            problems.append(f"unknown test '{test}'")
            continue

        ok = True
        for argument, needed in requirements.items():
            column = inner.get(argument)
            if not column:
                continue
            actual = column_types.get(str(column))
            if actual is None:
                continue
            if needed == "numeric" and actual != "numeric":
                ok = False
                problems.append(f"{test}: '{column}' is {actual}, needs numeric")
            if needed == "categorical" and actual not in {"categorical", "id", "boolean"}:
                ok = False
                problems.append(f"{test}: '{column}' is {actual}, needs categorical")

        # a two-sample t-test needs exactly two groups
        if test == "ttest":
            groups = inner.get("groups")
            if groups is not None and len(groups) != 2:
                ok = False
                problems.append(f"ttest given {len(groups)} groups, needs 2")

        appropriate += int(ok)

    return {
        "tests_run": len(calls),
        "tests_appropriate": appropriate,
        "test_appropriateness": round(appropriate / len(calls), 3),
        "test_problems": problems[:5],
    }


def recommendation_feedback(db) -> dict:
    """Adoption and usefulness from the feedback table (proposal Sec.20).

    This is the only metric here that depends on real users rather than injected
    ground truth, so it reports zero until somebody has actually rated something.
    """
    from app.models import Feedback

    rows = (db.query(Feedback)
            .filter(Feedback.target_type == "recommendation").all())
    if not rows:
        return {"feedback_count": 0, "adoption_rate": None, "mean_rating": None}

    adopted = [r for r in rows if r.was_adopted]
    ratings = [r.rating for r in rows if r.rating is not None]
    return {
        "feedback_count": len(rows),
        "adoption_rate": round(len(adopted) / len(rows), 3),
        "mean_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
    }
