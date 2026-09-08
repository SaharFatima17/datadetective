"""Data drift and join validation (proposal Sec.10).

Two checks that need more than one dataframe, so they live apart from the
single-frame profiling checks:

  drift  - compare column distributions between two dataset versions, to catch
           a schema or population change that would silently invalidate a
           comparison across periods
  join   - key uniqueness, unmatched rows and cardinality, checked BEFORE a
           merge rather than after, because a many-to-many join that quietly
           multiplies rows will corrupt every aggregate downstream
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Population Stability Index bands, the convention used in credit-risk work:
# below 0.1 no meaningful shift, 0.1-0.25 moderate, above 0.25 major.
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25


def _psi(previous: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    previous, current = previous.dropna(), current.dropna()
    if len(previous) < 10 or len(current) < 10:
        return 0.0

    edges = np.unique(np.quantile(previous, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    prev_pct = np.histogram(previous, bins=edges)[0] / len(previous)
    curr_pct = np.histogram(current, bins=edges)[0] / len(current)
    # floor the shares so an empty bucket does not send the index to infinity
    prev_pct = np.clip(prev_pct, 1e-6, None)
    curr_pct = np.clip(curr_pct, 1e-6, None)
    return float(np.sum((curr_pct - prev_pct) * np.log(curr_pct / prev_pct)))


def _categorical_shift(previous: pd.Series, current: pd.Series) -> dict:
    prev_share = previous.astype(str).value_counts(normalize=True)
    curr_share = current.astype(str).value_counts(normalize=True)
    labels = set(prev_share.index) | set(curr_share.index)

    biggest, biggest_label = 0.0, None
    for label in labels:
        delta = abs(float(curr_share.get(label, 0.0)) - float(prev_share.get(label, 0.0)))
        if delta > biggest:
            biggest, biggest_label = delta, label

    return {
        "max_share_change": round(biggest, 4),
        "changed_label": biggest_label,
        "new_labels": sorted(set(curr_share.index) - set(prev_share.index))[:10],
        "missing_labels": sorted(set(prev_share.index) - set(curr_share.index))[:10],
    }


def detect_drift(previous: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare two dataset versions column by column (proposal Sec.10)."""
    added = [str(c) for c in current.columns if c not in previous.columns]
    removed = [str(c) for c in previous.columns if c not in current.columns]
    shared = [c for c in current.columns if c in previous.columns]

    columns = []
    for column in shared:
        before, after = previous[column], current[column]
        entry: dict = {"column": str(column)}

        if str(before.dtype) != str(after.dtype):
            entry["type_changed"] = f"{before.dtype} -> {after.dtype}"

        if pd.api.types.is_numeric_dtype(before) and pd.api.types.is_numeric_dtype(after):
            psi = _psi(before, after)
            entry.update(
                psi=round(psi, 4),
                severity="high" if psi > PSI_MAJOR
                else "medium" if psi > PSI_MODERATE else "low",
                mean_before=round(float(before.mean()), 4) if before.notna().any() else None,
                mean_after=round(float(after.mean()), 4) if after.notna().any() else None,
            )
        else:
            shift = _categorical_shift(before.dropna(), after.dropna())
            severity = ("high" if shift["max_share_change"] > 0.25
                        else "medium" if shift["max_share_change"] > 0.10 else "low")
            entry.update(severity=severity, **shift)

        null_change = float(after.isna().mean() - before.isna().mean())
        if abs(null_change) > 0.05:
            entry["null_rate_change"] = round(null_change, 4)
            entry["severity"] = "high"

        columns.append(entry)

    drifted = [c for c in columns if c.get("severity") in {"high", "medium"}]
    return {
        "rows_before": len(previous),
        "rows_after": len(current),
        "columns_added": added,
        "columns_removed": removed,
        "drifted_columns": [c["column"] for c in drifted],
        "columns": sorted(columns,
                          key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(
                              c.get("severity", "low"), 2)),
        "verdict": (
            "major drift - the two versions are not directly comparable"
            if any(c.get("severity") == "high" for c in columns)
            else "moderate drift - interpret comparisons with care"
            if drifted else "no meaningful drift detected"
        ),
    }


# --------------------------------------------------------------------- #
def validate_join(left: pd.DataFrame, right: pd.DataFrame, on: str) -> dict:
    """Check a join key before merging (proposal Sec.10, join problems)."""
    if on not in left.columns:
        raise ValueError(f"Join key '{on}' is not in the base dataset")
    if on not in right.columns:
        raise ValueError(f"Join key '{on}' is not in the supplied dataset")

    left_key = left[on]
    right_key = right[on]
    left_unique = left_key.nunique(dropna=True) == len(left_key.dropna())
    right_unique = right_key.nunique(dropna=True) == len(right_key.dropna())

    cardinality = (
        "one-to-one" if left_unique and right_unique
        else "many-to-one" if right_unique
        else "one-to-many" if left_unique
        else "many-to-many"
    )

    left_values = set(left_key.dropna().astype(str))
    right_values = set(right_key.dropna().astype(str))
    matched = left_values & right_values

    problems = []
    if cardinality == "many-to-many":
        problems.append(
            "Many-to-many join: rows would be multiplied, corrupting every "
            "downstream sum and count."
        )
    if not matched:
        problems.append("No key values match between the two datasets.")
    elif len(matched) / max(len(left_values), 1) < 0.5:
        problems.append(
            f"Only {len(matched)}/{len(left_values)} base key values match; "
            "most rows would gain nothing from the join."
        )
    if left_key.isna().any() or right_key.isna().any():
        problems.append("The join key contains nulls, which never match.")

    return {
        "key": on,
        "cardinality": cardinality,
        "left_key_unique": bool(left_unique),
        "right_key_unique": bool(right_unique),
        "left_rows": len(left),
        "right_rows": len(right),
        "matched_keys": len(matched),
        "unmatched_left_keys": len(left_values - right_values),
        "unmatched_right_keys": len(right_values - left_values),
        "match_rate": round(len(matched) / max(len(left_values), 1), 4),
        "problems": problems,
        "safe": not problems,
    }


def suggest_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    """Pick a join key: a shared datetime column first, then a shared id-like one."""
    from app.services.profiling import infer_type

    shared = [c for c in left.columns if c in right.columns]
    if not shared:
        return None

    for column in shared:
        if (infer_type(left[column], str(column)) == "datetime"
                and infer_type(right[column], str(column)) == "datetime"):
            return str(column)

    best, best_score = None, 0.0
    for column in shared:
        if infer_type(left[column], str(column)) not in {"id", "categorical"}:
            continue
        left_values = set(left[column].dropna().astype(str))
        right_values = set(right[column].dropna().astype(str))
        if not left_values or not right_values:
            continue
        overlap = len(left_values & right_values) / len(left_values)
        uniqueness = right[column].nunique(dropna=True) / max(len(right), 1)
        score = overlap * (0.5 + 0.5 * uniqueness)
        if score > best_score:
            best, best_score = str(column), score

    return best if best_score >= 0.5 else None
