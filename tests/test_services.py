"""Unit tests for the deterministic layers (proposal Sec.16, Testing: Pytest).

These need no database and no API key - they test the parts that must be correct
regardless of which LLM is configured.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import analytics, cleaning, forecasting, profiling
from app.services.rag import chunk_text


# ---------------------------------------------------------------- profiling
def test_infers_column_types():
    df = pd.DataFrame({
        "order_id": [f"O{i:04d}" for i in range(100)],
        "order_date": pd.date_range("2024-01-01", periods=100).astype(str),
        "region": ["North", "South"] * 50,
        "revenue": np.random.default_rng(0).normal(500, 50, 100),
    })
    assert profiling.infer_type(df["order_id"], "order_id") == "id"
    assert profiling.infer_type(df["order_date"], "order_date") == "datetime"
    assert profiling.infer_type(df["region"], "region") == "categorical"
    assert profiling.infer_type(df["revenue"], "revenue") == "numeric"


def test_detects_missing_duplicates_and_negatives():
    df = pd.DataFrame({"revenue": [100.0, -20.0, None, 100.0] * 5})
    issues = profiling.column_quality_issues(df["revenue"], "revenue", "numeric")
    kinds = {i["issue"] for i in issues}
    assert "missing_values" in kinds
    assert "invalid_range" in kinds
    assert {i["issue"] for i in profiling.dataset_quality_issues(df)} == {"duplicate_rows"}


def test_health_score_falls_with_severity():
    clean = profiling._health_score([])
    bad = profiling._health_score([{"severity": "high"}, {"severity": "high"}])
    assert clean == 100 and bad < clean


def test_date_column_with_bad_values_is_still_a_date():
    """A quarter of unparseable values must not hide the column's real type."""
    series = pd.Series(["2024-01-01", "2024-01-02", "not-a-date", "2024-01-04"] * 10)
    assert profiling.infer_type(series, "order_date") == "datetime"
    kinds = {i["issue"] for i in profiling.column_quality_issues(series, "order_date", "datetime")}
    assert "date_problems" in kinds


# ---------------------------------------------------------------- cleaning
def test_destructive_operations_need_approval():
    df = pd.DataFrame({"revenue": [100.0, -50.0, None, 100.0] * 5,
                       "region": [" north", "North ", "SOUTH", "south"] * 5})
    plan = cleaning.propose_plan(df)
    assert plan["requires_approval_count"] > 0

    without, log = cleaning.apply_plan(df, plan, approved_op_ids=[])
    # nothing destructive ran, so the missing values are still missing
    assert all(o["safe"] for o in log if o.get("executed"))
    assert without["revenue"].isna().any()

    approved = [o["op_id"] for o in plan["operations"] if not o["safe"]]
    with_all, log2 = cleaning.apply_plan(df, plan, approved_op_ids=approved)
    assert any(o.get("executed") and not o["safe"] for o in log2)


def test_cleaning_never_mutates_the_input():
    df = pd.DataFrame({"revenue": [1.0, -1.0, None, 1.0]})
    before = df.copy()
    plan = cleaning.propose_plan(df)
    cleaning.apply_plan(df, plan, [o["op_id"] for o in plan["operations"]])
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------- analytics
def test_only_whitelisted_operations_run():
    df = pd.DataFrame({"a": [1, 2, 3]})
    for blocked in ("eval", "exec", "os.system", "to_csv"):
        with pytest.raises(ValueError):
            analytics.run_dataframe_op(df, blocked, {})
    # a whitelisted one still works
    assert analytics.run_dataframe_op(df, "describe", {"columns": ["a"]})["operation"] == "describe"


def test_period_contribution_finds_the_declining_group():
    """The group that caused the decline is not the largest group overall."""
    rows = []
    for month in pd.date_range("2024-01-01", periods=12, freq="MS"):
        for region, base in [("Big", 1000), ("Small", 100)]:
            value = base
            if region == "Small" and month >= pd.Timestamp("2024-10-01"):
                value = 10          # Small collapses; Big is unchanged
            rows.append({"date": month, "region": region, "revenue": value})
    df = pd.DataFrame(rows)

    out = analytics.run_dataframe_op(df, "period_contribution",
                                     {"date_column": "date", "metric": "revenue", "by": "region"})
    assert out["direction"] == "decline"
    assert out["result"][0]["group"] == "Small"
    assert out["result"][0]["contribution_to_total_change_pct"] > 90


def test_statistical_test_reports_effect_size_and_assumptions():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "value": np.concatenate([rng.normal(10, 1, 60), rng.normal(15, 1, 60)]),
        "group": ["a"] * 60 + ["b"] * 60,
    })
    r = analytics.statistical_test(df, "ttest", {"value_column": "value", "group_column": "group"})
    assert r["significant"] is True
    assert abs(r["cohens_d"]) > 0.8
    assert r["assumptions"]


def test_sql_is_read_only():
    for bad in ["DROP TABLE data", "DELETE FROM data", "UPDATE data SET x=1"]:
        with pytest.raises(ValueError):
            analytics.run_readonly_sql("dummy.parquet", bad)


# -------------------------------------------------------------- forecasting
def test_short_series_is_withheld_not_guessed():
    series = pd.Series([100, 110, 105, 120],
                       index=pd.date_range("2024-01-01", periods=4, freq="MS"))
    out = forecasting.forecast_series(series, horizon=3)
    assert out["reliability"] == "withheld_insufficient_data"
    assert out["predictions"] == []
    assert out["withheld_reason"]


def test_forecast_always_returns_a_range():
    rng = np.random.default_rng(2)
    values = 100 + np.arange(36) * 2 + rng.normal(0, 3, 36)
    series = pd.Series(values, index=pd.date_range("2022-01-01", periods=36, freq="MS"))
    out = forecasting.forecast_series(series, horizon=3)
    assert out["reliability"] in {"ok", "low_confidence"}
    for p in out["predictions"]:
        assert p["lower"] <= p["point"] <= p["upper"]
    assert out["backtest_metrics"]["folds"] > 0


def test_model_choice_depends_on_series_length():
    assert forecasting.select_model(6, False) == "insufficient"
    assert forecasting.select_model(15, False) == "ets"
    assert forecasting.select_model(40, True) == "sarima"


# --------------------------------------------------------------------- rag
def test_chunking_covers_the_whole_document():
    text = "\n\n".join(f"Paragraph {i} with some content in it." for i in range(40))
    chunks = chunk_text(text, size=300)
    assert len(chunks) > 1
    assert "Paragraph 0" in chunks[0]
    assert "Paragraph 39" in chunks[-1]
