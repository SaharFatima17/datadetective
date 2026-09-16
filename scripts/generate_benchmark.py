"""Phase 13 - Benchmark scenario generator (proposal Sec.21).

Generates synthetic datasets where the TRUE root cause is known in advance, so
root-cause ranking accuracy can actually be measured rather than eyeballed.

    python scripts/generate_benchmark.py

Writes CSVs plus a ground_truth.json into benchmarks/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "benchmarks"


def _dates(n_months: int = 24) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=n_months, freq="MS")


# --------------------------------------------------------------------- #
def scenario_regional_decline() -> tuple[pd.DataFrame, dict]:
    """True cause: one region collapses in the final quarter."""
    rows = []
    regions = ["North", "South", "East", "West"]
    for month in _dates():
        for region in regions:
            base = 50_000 + rng.normal(0, 3_000)
            # starts early enough to survive the forecast holdout, which trims
            # the final three months from the scored file
            if region == "South" and month >= pd.Timestamp("2024-07-01"):
                base *= 0.35                       # the injected cause
            for product in ["Alpha", "Beta", "Gamma"]:
                rows.append({
                    "date": month,
                    "region": region,
                    "product": product,
                    "revenue": round(max(0, base / 3 + rng.normal(0, 900)), 2),
                    "units": int(max(0, base / 300 + rng.normal(0, 12))),
                    "customer_id": f"C{rng.integers(1, 400):04d}",
                })
    return pd.DataFrame(rows), {
        "scenario": "regional_decline",
        "change_date": "2024-07-01",
        "true_cause": "Revenue decline is concentrated in region South",
        "true_driver_column": "region",
        "true_driver_value": "South",
        "target_metric": "revenue",
        "best_known_action": "Investigate and restore performance in the South region",
    }


def scenario_stock_shortage() -> tuple[pd.DataFrame, dict]:
    """True cause: availability drops, and revenue follows it."""
    rows = []
    for month in _dates():
        for product in ["Alpha", "Beta", "Gamma"]:
            availability = 0.95
            if product == "Beta" and month >= pd.Timestamp("2024-06-01"):
                availability = 0.45                # the injected cause
            base = 40_000 * availability + rng.normal(0, 1_500)
            rows.append({
                "date": month,
                "product": product,
                "region": rng.choice(["North", "South", "East", "West"]),
                "in_stock_rate": round(availability + rng.normal(0, 0.02), 3),
                "revenue": round(max(0, base), 2),
                "returns": int(max(0, rng.normal(20, 5))),
            })
    return pd.DataFrame(rows), {
        "scenario": "stock_shortage",
        "change_date": "2024-06-01",
        "true_cause": "in_stock_rate correlates with revenue; Beta availability fell",
        "true_driver_column": "in_stock_rate",
        "true_driver_value": "Beta",
        "magnitude_group_column": "product",
        "target_metric": "revenue",
        "best_known_action": "Restock product Beta to restore availability",
    }


def scenario_price_increase() -> tuple[pd.DataFrame, dict]:
    """True cause: a price rise suppresses units; revenue follows."""
    rows = []
    for month in _dates():
        price = 100.0
        if month >= pd.Timestamp("2024-07-01"):
            price = 145.0                          # the injected cause
        for _ in range(30):
            # elastic demand: units fall faster than price rises, so revenue drops
            units = max(0, int(60 * (100 / price) ** 2.2 + rng.normal(0, 3)))
            rows.append({
                "date": month,
                "price": round(price + rng.normal(0, 2), 2),
                "units": units,
                "revenue": round(units * price, 2),
                "channel": rng.choice(["online", "retail"]),
            })
    return pd.DataFrame(rows), {
        "scenario": "price_increase",
        "change_date": "2024-07-01",
        "true_cause": "price is inversely related to units, reducing revenue",
        "true_driver_column": "price",
        "true_driver_value": None,
        "target_metric": "revenue",
        "best_known_action": "Review the price increase introduced in July 2024",
    }


def scenario_missing_evidence() -> tuple[pd.DataFrame, dict]:
    """No column can explain the drop - the system SHOULD ask instead of guessing."""
    rows = []
    for month in _dates():
        base = 60_000 if month < pd.Timestamp("2024-09-01") else 38_000
        for _ in range(20):
            rows.append({
                "date": month,
                "revenue": round(max(0, base / 20 + rng.normal(0, 300)), 2),
                "order_id": f"O{rng.integers(1, 9999):05d}",
            })
    return pd.DataFrame(rows), {
        "scenario": "missing_evidence",
        "change_date": "2024-09-01",
        "true_cause": "Not determinable from this dataset - an external factor",
        "true_driver_column": None,
        "true_driver_value": None,
        "target_metric": "revenue",
        "expected_behaviour": "should raise a missing-evidence request, not assert a cause",
        "best_known_action": "Request additional operational data before concluding",
    }


def scenario_multi_source() -> tuple[pd.DataFrame, dict]:
    """True cause is only fully explainable by combining the table AND a document.

    The table shows WHERE revenue fell (North / Widget). Nothing in the table
    says why - that is only in the incident report. A system without retrieval
    can find the segment but cannot name the cause.
    """
    rows = []
    for month in _dates():
        for region in ["North", "South", "East"]:
            for product in ["Widget", "Gadget"]:
                base = 30_000 + rng.normal(0, 1_500)
                # starts early enough that the effect survives the forecast
                # holdout, which trims the final three months from the file
                if (region == "North" and product == "Widget"
                        and month >= pd.Timestamp("2024-04-01")):
                    base *= 0.25              # the injected effect
                rows.append({
                    "date": month,
                    "region": region,
                    "product": product,
                    "revenue": round(max(0, base), 2),
                    "orders": int(max(0, base / 200 + rng.normal(0, 8))),
                })

    document = {
        "title": "Warehouse incident report - North distribution centre",
        "document_type": "business_doc",
        "text": (
            "Warehouse incident report - North distribution centre\n\n"
            "On 2 April 2024 a cooling system failure at the North "
            "distribution centre forced the closure of aisle 4, where all Widget "
            "stock is held. Widget inventory available for dispatch in the North "
            "region fell to roughly one quarter of normal levels and has not yet "
            "been restored.\n\n"
            "Impact: Widget orders originating in the North region could not be "
            "fulfilled from April 2024 onward. Other products and other "
            "regions were unaffected because their stock is held in separate "
            "aisles.\n\n"
            "Remediation: the cooling unit replacement is scheduled but not yet "
            "complete. Until then, Widget availability in the North region "
            "remains constrained."
        ),
    }

    return pd.DataFrame(rows), {
        "scenario": "multi_source",
        "change_date": "2024-04-01",
        "true_cause": (
            "Widget revenue in the North region fell because a warehouse cooling "
            "failure cut Widget stock availability"
        ),
        "true_driver_column": "region",
        "true_driver_value": "North",
        "target_metric": "revenue",
        "requires_document": True,
        "document_evidence_terms": ["warehouse", "cooling", "stock", "aisle"],
        "best_known_action": "Restore Widget stock availability at the North distribution centre",
        "document": document,
    }


def scenario_changed_driver() -> tuple[pd.DataFrame, dict]:
    """The driver in the later period differs from the earlier one.

    Period 1 (2023): Region A collapses.  Period 2 (2024): Region A has
    recovered and Product Y collapses instead. A system that only remembers its
    previous answer will report the old driver.
    """
    rows = []
    for month in pd.date_range("2023-01-01", periods=36, freq="MS"):
        for region in ["A", "B", "C"]:
            for product in ["X", "Y"]:
                base = 20_000 + rng.normal(0, 800)
                # first decline: region A, mid-2023
                if region == "A" and pd.Timestamp("2023-06-01") <= month < pd.Timestamp("2024-01-01"):
                    base *= 0.30
                # second decline: product Y, late 2024, region A back to normal
                if product == "Y" and month >= pd.Timestamp("2024-09-01"):
                    base *= 0.30
                rows.append({
                    "date": month,
                    "region": region,
                    "product": product,
                    "revenue": round(max(0, base), 2),
                })

    return pd.DataFrame(rows), {
        "scenario": "changed_driver",
        "change_date": "2024-09-01",
        "true_cause": "The current decline is driven by product Y, not by region A as before",
        "true_driver_column": "product",
        "true_driver_value": "Y",
        "previous_driver_column": "region",
        "previous_driver_value": "A",
        "target_metric": "revenue",
        "tests_driver_change": True,
        "earlier_scope": {"start": "2023-01-01", "end": "2023-12-31"},
        "scope": {"start": "2024-01-01", "end": "2025-12-31"},
        "best_known_action": "Address the product Y decline; the earlier region A problem has resolved",
    }


def scenario_dirty_data() -> tuple[pd.DataFrame, dict]:
    """A dataset with a known list of injected defects (proposal Sec.20).

    Every defect below is recorded in `expected_issues` as (column, issue), which
    is what makes data-quality detection precision/recall measurable rather than
    a matter of opinion. A clean reference copy is written alongside it so
    cleaning can be scored on whether it repaired the defects WITHOUT destroying
    valid rows.
    """
    n = 240
    clean = pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(n)],
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "revenue": np.round(rng.normal(500, 60, n), 2),
        "units": rng.integers(1, 20, n),
    })

    dirty = clean.copy()
    dirty["order_date"] = dirty["order_date"].dt.strftime("%Y-%m-%d")

    # 1. missing values in revenue
    dirty.loc[dirty.index[:18], "revenue"] = np.nan
    # 2. negative revenue (invalid range)
    dirty.loc[dirty.index[20:26], "revenue"] = -100.0
    # 3. extreme outliers
    dirty.loc[dirty.index[30:33], "revenue"] = 90_000.0
    # 4. inconsistent category casing and spacing
    dirty.loc[dirty.index[40:60], "region"] = " north "
    # 5. a probable typo variant of an existing label
    dirty.loc[dirty.index[60:66], "region"] = "Norht"
    # 6. unparseable dates
    dirty.loc[dirty.index[70:75], "order_date"] = "not-a-date"
    # 7. duplicate identifiers (rows differ, so exact-duplicate checks miss them)
    dirty.loc[dirty.index[80:86], "order_id"] = "O00001"
    # 8. exact duplicate rows
    dirty = pd.concat([dirty, dirty.iloc[100:110]], ignore_index=True)

    expected = [
        ["revenue", "missing_values"],
        ["revenue", "invalid_range"],
        ["revenue", "outliers"],
        ["region", "inconsistent_categories"],
        ["region", "similar_categories"],
        ["order_date", "date_problems"],
        ["order_id", "duplicate_keys"],
        ["__dataset__", "duplicate_rows"],
    ]

    return dirty, {
        "scenario": "dirty_data",
        "true_cause": "Not a root-cause scenario - this one scores profiling and cleaning",
        "true_driver_column": None,
        "true_driver_value": None,
        "target_metric": "revenue",
        "quality_benchmark": True,
        "expected_issues": expected,
        # Defects the system deliberately does NOT repair automatically.
        # Detecting them is required; silently removing them is not:
        #   similar_categories - merging on similarity alone destroys real
        #                        distinctions, so it is a suggestion only
        #   duplicate_keys     - deduplicating needs to know which row is
        #                        authoritative, which only the user knows
        #   outliers           - an outlier is often the thing being
        #                        investigated; deleting it would remove the
        #                        very signal this system exists to find
        # Scored separately so "not resolved" is not confused with "missed".
        "not_auto_repairable": [
            ["region", "similar_categories"],
            ["order_id", "duplicate_keys"],
            ["revenue", "outliers"],
        ],
        "clean_row_count": n,
        "injected_duplicate_rows": 10,
        "best_known_action": "Clean the dataset before drawing conclusions from it",
    }


SCENARIOS = [
    scenario_regional_decline,
    scenario_stock_shortage,
    scenario_price_increase,
    scenario_missing_evidence,
    scenario_multi_source,
    scenario_changed_driver,
    scenario_dirty_data,
]


def compute_true_magnitude(df: pd.DataFrame, meta: dict) -> float | None:
    """The per-period change for the injected driver group (proposal Sec.20).

    Written independently of app/services so it is a genuine check on the
    system's arithmetic rather than a restatement of it.

    The split is the date this scenario actually injected the change, which the
    generator knows because it put it there. Earlier this mirrored the system's
    own heuristic - the most recent third of the range - which meant the check
    moved whenever that heuristic moved, and measured agreement with a guess
    rather than agreement with the truth. Anchoring it to the injected date
    makes the comparison independent of how the system chooses its split.
    """
    # some scenarios name the mechanism in true_driver_column (e.g. in_stock_rate)
    # while the segment the system reports on is a different column
    column = meta.get("magnitude_group_column") or meta.get("true_driver_column")
    value = meta.get("true_driver_value")
    metric = meta.get("target_metric")
    if not (column and value and metric) or column not in df.columns:
        return None
    if "date" not in df.columns:
        return None
    if str(value) not in set(df[column].astype(str)):
        return None

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    # score inside the same window the investigation will be scoped to
    scope = meta.get("scope") or {}
    if scope.get("start"):
        frame = frame[frame["date"] >= pd.Timestamp(scope["start"])]
    if scope.get("end"):
        frame = frame[frame["date"] <= pd.Timestamp(scope["end"])]
    if frame.empty:
        return None
    change_date = meta.get("change_date")
    cutoff = (pd.Timestamp(change_date) if change_date
              else frame["date"].quantile(0.67))
    previous = frame[frame["date"] < cutoff]
    current = frame[frame["date"] >= cutoff]
    if previous.empty or current.empty:
        return None

    prev_periods = max(previous["date"].dt.to_period("M").nunique(), 1)
    curr_periods = max(current["date"].dt.to_period("M").nunique(), 1)

    prev_total = previous[previous[column].astype(str) == str(value)][metric].sum()
    curr_total = current[current[column].astype(str) == str(value)][metric].sum()
    return abs(float(curr_total / curr_periods - prev_total / prev_periods))


def withhold_tail(df: pd.DataFrame, meta: dict, periods: int = 3) -> dict:
    """Hold back the final periods as forecast ground truth (proposal Sec.21).

    The scored file stops before these periods; the actuals are recorded in the
    ground truth so a forecast can be checked against what really happened
    rather than only against its own backtest.
    """
    if ("date" not in df.columns or meta.get("target_metric") != "revenue"
            or meta.get("quality_benchmark")):
        return {}

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    monthly = frame.groupby(frame["date"].dt.to_period("M"))["revenue"].sum()
    if len(monthly) <= periods + 12:
        return {}

    held = monthly.tail(periods)
    cutoff = held.index[0].to_timestamp()
    return {
        "cutoff": str(cutoff.date()),
        "held_out_periods": [
            {"period": str(p.to_timestamp().date()), "actual": round(float(v), 2)}
            for p, v in held.items()
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = []
    for fn in SCENARIOS:
        df, meta = fn()

        holdout = withhold_tail(df, meta)
        if holdout:
            frame = df.copy()
            frame["date"] = pd.to_datetime(frame["date"])
            visible = frame[frame["date"] < pd.Timestamp(holdout["cutoff"])]
            # only withhold when plenty of history remains to forecast from
            if len(visible) > 0.6 * len(frame):
                df = visible
                meta["forecast_holdout"] = holdout

        magnitude = compute_true_magnitude(df, meta)
        if magnitude is not None:
            meta["true_magnitude"] = round(magnitude, 2)

        path = OUT / f"{meta['scenario']}.csv"
        df.to_csv(path, index=False)
        meta["file"] = path.name
        meta["rows"] = len(df)

        if meta.get("quality_benchmark"):
            reference = OUT / f"{meta['scenario']}_reference.csv"
            # the defect-free version, for scoring what cleaning preserved
            ref = df.drop_duplicates().copy()
            meta["reference_file"] = reference.name
            ref.to_csv(reference, index=False)

        document = meta.pop("document", None)
        if document:
            doc_path = OUT / f"{meta['scenario']}_context.txt"
            doc_path.write_text(document["text"])
            meta["document"] = {**document, "file": doc_path.name}

        truth.append(meta)
        print(f"  {path.name:28s} {len(df):6d} rows   true cause: {meta['true_cause']}")

    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"\nWrote {len(truth)} scenarios and ground_truth.json to {OUT}")


if __name__ == "__main__":
    main()