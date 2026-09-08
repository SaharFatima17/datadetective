"""Phase 4 - Cleaning (proposal Sec.7 steps 8-9, Sec.10).

Two-step by design:
  1. propose_plan  - read-only, explains every operation and its estimated impact
  2. apply_plan    - executes; safe ops run automatically, destructive ops need
                     explicit approval (Sec.19), and a NEW version is always
                     created rather than overwriting the input

Operation order matters and is fixed here, because e.g. nulling negatives before
imputing means those nulls get imputed too. That interaction is visible in the
difference between estimated_affected_rows and actual_affected_rows.
"""

from __future__ import annotations

import pandas as pd

from app.services.profiling import AMOUNT_HINTS, infer_type, parse_datetimes

# ops that cannot lose information
SAFE_OPS = {"drop_exact_duplicates", "normalize_category_casing", "strip_whitespace"}
# ops that change or remove real values - require approval
DESTRUCTIVE_OPS = {
    "coerce_datetime",
    "null_out_negative_values",
    "impute_missing_median",
    "impute_missing_mode",
    "drop_constant_columns",
}
# fixed execution order
OP_ORDER = [
    "strip_whitespace",
    "normalize_category_casing",
    "drop_exact_duplicates",
    "coerce_datetime",
    "null_out_negative_values",
    "impute_missing_median",
    "impute_missing_mode",
    "drop_constant_columns",
]


def propose_plan(df: pd.DataFrame) -> dict:
    """Read-only. Returns the operations that would be applied and why."""
    ops: list[dict] = []

    dupes = int(df.duplicated().sum())
    if dupes:
        ops.append(
            {
                "op_id": "drop_exact_duplicates",
                "operation": "drop_exact_duplicates",
                "column": None,
                "safe": True,
                "reason": "Identical rows counted more than once distort every aggregate.",
                "estimated_affected_rows": dupes,
            }
        )

    for name in df.columns:
        series = df[name]
        col = str(name)
        inferred = infer_type(series, col)
        non_null = series.dropna()

        if inferred in {"categorical", "text"} and len(non_null):
            as_str = non_null.astype(str)
            if (as_str != as_str.str.strip()).any():
                ops.append(
                    {
                        "op_id": f"strip_whitespace::{col}",
                        "operation": "strip_whitespace",
                        "column": col,
                        "safe": True,
                        "reason": "Leading/trailing spaces split otherwise identical labels.",
                        "estimated_affected_rows": int((as_str != as_str.str.strip()).sum()),
                    }
                )
            stripped = as_str.str.strip()
            if stripped.str.lower().nunique() < stripped.nunique():
                ops.append(
                    {
                        "op_id": f"normalize_category_casing::{col}",
                        "operation": "normalize_category_casing",
                        "column": col,
                        "safe": True,
                        "reason": (
                            f"{stripped.nunique()} labels collapse to "
                            f"{stripped.str.lower().nunique()} once case is normalised."
                        ),
                        "estimated_affected_rows": int(len(stripped)),
                    }
                )

        if inferred == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
            # pandas 2 stores text as `object`, pandas 3 stores it as `str`.
            # Testing for "not already a datetime" instead of "is object" keeps
            # this working on both.
            parsed = parse_datetimes(non_null)
            ops.append(
                {
                    "op_id": f"coerce_datetime::{col}",
                    "operation": "coerce_datetime",
                    "column": col,
                    "safe": False,
                    "reason": (
                        "Stored as text; converting enables time-based analysis. "
                        f"{int(parsed.isna().sum())} unparseable values become null."
                    ),
                    "estimated_affected_rows": int(len(non_null)),
                }
            )

        if inferred == "numeric" and any(h in col.lower() for h in AMOUNT_HINTS):
            negatives = int((non_null < 0).sum())
            if negatives:
                ops.append(
                    {
                        "op_id": f"null_out_negative_values::{col}",
                        "operation": "null_out_negative_values",
                        "column": col,
                        "safe": False,
                        "reason": "Negative values are implausible for this measure; they may be data-entry errors.",
                        "estimated_affected_rows": negatives,
                    }
                )

        missing = int(series.isna().sum())
        if missing and len(df):
            pct = missing / len(df) * 100
            if pct < 40:
                if inferred == "numeric":
                    ops.append(
                        {
                            "op_id": f"impute_missing_median::{col}",
                            "operation": "impute_missing_median",
                            "column": col,
                            "safe": False,
                            "reason": f"{pct:.1f}% missing; median imputation is robust to outliers.",
                            "estimated_affected_rows": missing,
                        }
                    )
                elif inferred == "categorical":
                    ops.append(
                        {
                            "op_id": f"impute_missing_mode::{col}",
                            "operation": "impute_missing_mode",
                            "column": col,
                            "safe": False,
                            "reason": f"{pct:.1f}% missing; filled with the most frequent label.",
                            "estimated_affected_rows": missing,
                        }
                    )

        if len(non_null) and non_null.nunique() == 1:
            ops.append(
                {
                    "op_id": f"drop_constant_columns::{col}",
                    "operation": "drop_constant_columns",
                    "column": col,
                    "safe": False,
                    "reason": "Column has a single value throughout and cannot explain any variation.",
                    "estimated_affected_rows": int(len(df)),
                }
            )

    ops.sort(key=lambda o: OP_ORDER.index(o["operation"]))
    return {
        "operations": ops,
        "safe_count": sum(1 for o in ops if o["safe"]),
        "requires_approval_count": sum(1 for o in ops if not o["safe"]),
        "note": (
            "Safe operations run automatically. Destructive operations run only if "
            "their op_id is listed in approved_op_ids."
        ),
    }


def apply_plan(
    df: pd.DataFrame, plan: dict, approved_op_ids: list[str] | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """Applies the plan and returns (new_dataframe, executed_operation_log)."""
    approved = set(approved_op_ids or [])
    out = df.copy()
    log: list[dict] = []

    for op in plan["operations"]:
        if not op["safe"] and op["op_id"] not in approved:
            log.append({**op, "executed": False, "skipped_reason": "not approved"})
            continue

        before = out.copy()
        col = op["column"]
        name = op["operation"]

        try:
            if name == "drop_exact_duplicates":
                out = out.drop_duplicates()
                affected = len(before) - len(out)

            elif name == "strip_whitespace":
                mask = out[col].notna()
                out.loc[mask, col] = out.loc[mask, col].astype(str).str.strip()
                affected = int(mask.sum())

            elif name == "normalize_category_casing":
                mask = out[col].notna()
                out.loc[mask, col] = out.loc[mask, col].astype(str).str.strip().str.title()
                affected = int(mask.sum())

            elif name == "coerce_datetime":
                out[col] = parse_datetimes(out[col])
                affected = int(out[col].notna().sum())

            elif name == "null_out_negative_values":
                mask = out[col] < 0
                out.loc[mask, col] = None
                affected = int(mask.sum())

            elif name == "impute_missing_median":
                mask = out[col].isna()
                median = out[col].median()
                out[col] = out[col].fillna(median)
                affected = int(mask.sum())

            elif name == "impute_missing_mode":
                mask = out[col].isna()
                modes = out[col].mode()
                if len(modes):
                    out[col] = out[col].fillna(modes[0])
                affected = int(mask.sum())

            elif name == "drop_constant_columns":
                out = out.drop(columns=[col])
                affected = len(out)

            else:
                log.append({**op, "executed": False, "skipped_reason": "unknown operation"})
                continue

            log.append({**op, "executed": True, "actual_affected_rows": int(affected)})

        except Exception as exc:  # noqa: BLE001
            out = before
            log.append({**op, "executed": False, "skipped_reason": f"error: {exc}"})

    return out, log
