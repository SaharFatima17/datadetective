"""Tests for the features completing proposal Sec.10, Sec.12 and Sec.14."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import dataquality, forecasting, profiling


# ------------------------------------------------------- Sec.10 drift
def test_drift_detects_a_shifted_numeric_distribution():
    rng = np.random.default_rng(0)
    before = pd.DataFrame({"revenue": rng.normal(100, 10, 500)})
    after = pd.DataFrame({"revenue": rng.normal(160, 10, 500)})

    report = dataquality.detect_drift(before, after)
    entry = report["columns"][0]
    assert entry["psi"] > dataquality.PSI_MAJOR
    assert entry["severity"] == "high"
    assert "revenue" in report["drifted_columns"]
    assert "not directly comparable" in report["verdict"]


def test_drift_stays_quiet_when_nothing_changed():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"revenue": rng.normal(100, 10, 400)})
    report = dataquality.detect_drift(frame, frame.copy())
    assert report["drifted_columns"] == []
    assert "no meaningful drift" in report["verdict"]


def test_drift_reports_schema_changes_and_new_labels():
    before = pd.DataFrame({"region": ["North"] * 50, "revenue": [1.0] * 50})
    after = pd.DataFrame({"region": ["North"] * 25 + ["Mars"] * 25,
                          "orders": [2] * 50})
    report = dataquality.detect_drift(before, after)
    assert report["columns_added"] == ["orders"]
    assert report["columns_removed"] == ["revenue"]
    region = next(c for c in report["columns"] if c["column"] == "region")
    assert "Mars" in region["new_labels"]


# -------------------------------------------------------- Sec.10 joins
def test_many_to_many_join_is_refused():
    left = pd.DataFrame({"k": ["a", "a", "b"], "x": [1, 2, 3]})
    right = pd.DataFrame({"k": ["a", "a", "b"], "y": [4, 5, 6]})
    check = dataquality.validate_join(left, right, "k")
    assert check["cardinality"] == "many-to-many"
    assert check["safe"] is False
    assert any("multiplied" in p for p in check["problems"])


def test_clean_many_to_one_join_is_allowed():
    left = pd.DataFrame({"k": ["a", "a", "b"], "x": [1, 2, 3]})
    right = pd.DataFrame({"k": ["a", "b"], "y": [4, 5]})
    check = dataquality.validate_join(left, right, "k")
    assert check["cardinality"] == "many-to-one"
    assert check["safe"] is True
    assert check["match_rate"] == 1.0


def test_no_overlapping_keys_is_refused():
    left = pd.DataFrame({"k": ["a", "b"], "x": [1, 2]})
    right = pd.DataFrame({"k": ["y", "z"], "y": [3, 4]})
    assert dataquality.validate_join(left, right, "k")["safe"] is False


def test_join_key_suggestion_prefers_a_shared_date():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    left = pd.DataFrame({"date": dates, "sku": [f"S{i}" for i in range(20)], "x": 1})
    right = pd.DataFrame({"date": dates, "promo": range(20)})
    assert dataquality.suggest_join_key(left, right) == "date"


def test_no_shared_column_yields_no_suggestion():
    left = pd.DataFrame({"a": [1, 2]})
    right = pd.DataFrame({"b": [3, 4]})
    assert dataquality.suggest_join_key(left, right) is None


# ------------------------------------------- Sec.10 units and near-duplicates
def test_mixed_units_are_flagged():
    values = pd.Series(["5kg", "11 lb", "7kg", "3 lb"] * 5)
    issues = profiling.column_quality_issues(values, "weight", "categorical")
    unit_issue = next((i for i in issues if i["issue"] == "inconsistent_units"), None)
    assert unit_issue is not None
    assert unit_issue["severity"] == "high"
    assert set(unit_issue["units_found"]) == {"kg", "lb"}


def test_probable_typos_are_suggested_not_merged():
    values = pd.Series(["north", "norht", "south"] * 8)
    issues = profiling.column_quality_issues(values, "region", "categorical")
    issue = next((i for i in issues if i["issue"] == "similar_categories"), None)
    assert issue is not None
    assert issue["severity"] == "low"          # a suggestion, never automatic
    assert "will not merge" in issue["detail"]


def test_distinct_labels_are_not_flagged_as_typos():
    values = pd.Series(["north", "south", "east", "west"] * 8)
    issues = profiling.column_quality_issues(values, "region", "categorical")
    assert not any(i["issue"] == "similar_categories" for i in issues)


def test_robust_zscore_catches_an_extreme_value():
    values = pd.Series([10.0] * 40 + [9000.0])
    issues = profiling.column_quality_issues(values, "revenue", "numeric")
    assert any(i["issue"] == "extreme_outliers" for i in issues)
    assert any(i.get("method") == "robust_zscore" for i in issues)


# --------------------------------------------------- Sec.12 extra models
def test_gradient_boosting_is_chosen_when_external_features_exist():
    assert forecasting.select_model(30, False, has_external_features=True) == "gbr"
    assert forecasting.select_model(30, False, has_external_features=False) != "gbr"


def test_prophet_only_when_installed_and_trend_actually_changes():
    chosen = forecasting.select_model(36, False, trend_change=True)
    if forecasting.prophet_available():
        assert chosen == "prophet"
    else:
        assert chosen == "arima"          # falls back rather than failing
    assert forecasting.select_model(36, False, trend_change=False) == "arima"


def test_trend_change_detection():
    up_then_down = np.concatenate([np.arange(18) * 5.0, 90 - np.arange(18) * 5.0])
    steady = np.arange(36) * 5.0
    assert forecasting.has_trend_change(up_then_down) is True
    assert forecasting.has_trend_change(steady) is False


def test_gbr_forecast_produces_a_usable_range():
    rng = np.random.default_rng(3)
    values = 1000 + np.arange(30) * 5 + rng.normal(0, 40, 30)
    series = pd.Series(values, index=pd.date_range("2022-01-01", periods=30, freq="MS"))
    exog = pd.DataFrame({"price": 100 + rng.normal(0, 5, 30)}, index=series.index)

    out = forecasting.forecast_series(series, horizon=3, exog=exog)
    assert out["model_name"] == "gbr"
    assert out["external_features"] == ["price"]
    for p in out["predictions"]:
        assert p["lower"] < p["point"] < p["upper"]
    # bands built from in-sample residuals collapse to nothing; these must not
    width = out["predictions"][0]["upper"] - out["predictions"][0]["lower"]
    assert width > 0.001 * abs(out["predictions"][0]["point"])


# ------------------------------------------------ Sec.7 supplementary merge
def test_join_validation_blocks_an_unsafe_merge():
    base = pd.DataFrame({"k": ["a", "a", "b"], "revenue": [1.0, 2.0, 3.0]})
    supplementary = pd.DataFrame({"k": ["a", "a"], "stock": [1, 2]})
    check = dataquality.validate_join(base, supplementary, "k")
    assert check["safe"] is False


def test_merge_requires_new_columns(monkeypatch):
    """A file that adds nothing cannot unblock a hypothesis."""
    from app.services import ingestion

    base = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20),
                         "revenue": range(20)})
    supplementary = base.copy()

    class _Version:
        id = "v1"

    monkeypatch.setattr(ingestion, "load_version", lambda _v: base)
    with pytest.raises(ValueError, match="adds no new columns"):
        ingestion.merge_supplementary(None, None, _Version(), supplementary,
                                      join_on="date")


def test_merge_reports_a_missing_join_key(monkeypatch):
    from app.services import ingestion

    base = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20),
                         "revenue": range(20)})
    unrelated = pd.DataFrame({"something_else": range(5), "value": range(5)})

    class _Version:
        id = "v1"

    monkeypatch.setattr(ingestion, "load_version", lambda _v: base)
    with pytest.raises(ValueError, match="No usable join key"):
        ingestion.merge_supplementary(None, None, _Version(), unrelated)


# ------------------------------------------- Sec.10 completed detections
def test_key_duplicates_are_caught_when_rows_differ():
    """Exact-duplicate checks pass here; the identifier is still not unique."""
    df = pd.DataFrame({"order_id": ["A1", "A1", "A2", "A3"],
                       "revenue": [1.0, 2.0, 3.0, 4.0]})
    issues = profiling.dataset_quality_issues(df)
    assert not any(i["issue"] == "duplicate_rows" for i in issues)
    key_issue = next(i for i in issues if i["issue"] == "duplicate_keys")
    assert key_issue["severity"] == "high"
    assert key_issue["column"] == "order_id"


def test_unique_keys_are_not_flagged():
    df = pd.DataFrame({"order_id": ["A1", "A2", "A3"], "x": [1, 2, 3]})
    assert not any(i["issue"] == "duplicate_keys"
                   for i in profiling.dataset_quality_issues(df))


def test_mixed_timezones_are_reported_not_crashed():
    """pandas raises on mixed offsets; profiling must survive and report it."""
    values = pd.Series(["2024-01-01T00:00:00+05:00", "2024-01-02T00:00:00Z",
                        "2024-01-03T00:00:00"] * 8)
    assert profiling.parse_datetimes(values).notna().all()
    issues = profiling.column_quality_issues(values, "created_at", "datetime")
    assert any(i["issue"] == "timezone_inconsistency" for i in issues)


def test_out_of_order_dates_are_flagged():
    values = pd.Series(pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"] * 8))
    assert any(i["issue"] == "date_ordering"
               for i in profiling.column_quality_issues(values, "order_date", "datetime"))


def test_isolation_detection_stays_quiet_on_a_clean_table():
    rng = np.random.default_rng(7)
    clean = pd.DataFrame({"revenue": rng.normal(500, 50, 300),
                          "units": rng.integers(1, 20, 300)})
    assert not any(i["issue"] == "anomalous_rows"
                   for i in profiling.dataset_quality_issues(clean))


def test_isolation_detection_catches_a_row_odd_only_in_combination():
    """Each value is plausible alone; together they are not - which is exactly
    what the per-column IQR and z-score checks cannot see."""
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"revenue": rng.normal(500, 50, 300),
                       "units": rng.integers(1, 20, 300)})
    df.loc[0, ["revenue", "units"]] = [9000.0, 1]

    issues = profiling.dataset_quality_issues(df)
    anomaly = next(i for i in issues if i["issue"] == "anomalous_rows")
    assert anomaly["method"] == "isolation_forest"
    assert anomaly["affected_rows"] >= 1
    assert set(anomaly["columns_considered"]) == {"revenue", "units"}


# ------------------------------------------------ Sec.20 quality metrics
def test_quality_detection_scores_recall_and_precision():
    from app.evaluation import quality_metrics as qm

    truth = {"expected_issues": [["revenue", "missing_values"],
                                 ["revenue", "outliers"],
                                 ["region", "inconsistent_categories"]]}
    profile = {
        "dataset_issues": [],
        "columns": [
            {"name": "revenue", "issues": [{"issue": "missing_values"},
                                           {"issue": "extreme_outliers"}]},
            {"name": "region", "issues": []},
        ],
    }
    scored = qm.quality_detection(truth, profile)
    # extreme_outliers counts as the outliers defect
    assert scored["quality_detected"] == 2
    assert scored["quality_recall"] == round(2 / 3, 3)
    assert scored["quality_missed"] == ["region/inconsistent_categories"]


def test_cleaning_that_drops_everything_scores_badly():
    from app.evaluation import quality_metrics as qm

    truth = {"expected_issues": [["revenue", "missing_values"]],
             "clean_row_count": 100}
    before = pd.DataFrame({"revenue": [1.0] * 110})
    after = pd.DataFrame({"revenue": [1.0] * 5})       # over-cleaned
    scored = qm.cleaning_correctness(truth, before, after,
                                     {"dataset_issues": [], "columns": []})
    assert scored["defect_resolution_rate"] == 1.0
    assert scored["rows_over_cleaned"] == 95
    assert scored["cleaning_correct"] is False


def test_inappropriate_statistical_test_is_caught():
    from app.evaluation import quality_metrics as qm

    run = {"tool_runs": [
        {"tool_name": "run_statistical_test",
         "parameters": {"test": "correlation",
                        "params": {"column_a": "region", "column_b": "revenue"}}},
        {"tool_name": "run_statistical_test",
         "parameters": {"test": "ttest",
                        "params": {"value_column": "revenue",
                                   "group_column": "region",
                                   "groups": ["North", "South"]}}},
    ]}
    scored = qm.test_appropriateness(run, {"region": "categorical",
                                           "revenue": "numeric"})
    assert scored["tests_run"] == 2
    assert scored["tests_appropriate"] == 1      # correlation on a category
    assert any("needs numeric" in p for p in scored["test_problems"])


def test_numerical_accuracy_compares_against_the_injected_value():
    from app.evaluation import quality_metrics as qm

    truth = {"true_magnitude": 1000.0}
    exact = {"findings": [{"finding_type": "driver", "magnitude": 1000.0}]}
    wrong = {"findings": [{"finding_type": "driver", "magnitude": 400.0}]}
    assert qm.numerical_accuracy(truth, exact)["numerical_correct"] is True
    assert qm.numerical_accuracy(truth, wrong)["numerical_correct"] is False


def test_forecast_scoring_reports_rmse_and_coverage():
    from app.evaluation import metrics

    truth = {"forecast_holdout": {"held_out_periods": [
        {"period": "2025-01-01", "actual": 100.0},
        {"period": "2025-02-01", "actual": 110.0}]}}
    run = {"forecasts": [{"predictions": [
        {"point": 102.0, "lower": 90.0, "upper": 115.0},
        {"point": 108.0, "lower": 95.0, "upper": 120.0}]}]}
    scored = metrics.forecast_accuracy(truth, run)
    assert scored["forecast_scored"] is True
    assert scored["holdout_rmse"] is not None
    assert scored["interval_coverage"] == 1.0
