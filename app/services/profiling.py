"""Phase 3 - Profiling (proposal Sec.7 steps 5-7, Sec.10).

Everything here is deterministic pandas. No LLM touches these numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models import DatasetColumn, DatasetVersion

ID_HINTS = ("id", "code", "key", "uuid", "number", "no")
# unit tokens that may appear inside otherwise-numeric text values
UNIT_TOKENS = ("kg", "g", "lb", "lbs", "oz", "ton", "tonne", "km", "m", "cm", "mi",
               "l", "ml", "hr", "hrs", "min", "sec", "usd", "eur", "gbp", "pkr",
               "inr", "%", "$", "€", "£")
AMOUNT_HINTS = ("revenue", "sales", "amount", "price", "cost", "value", "total", "qty",
                "quantity", "count", "profit", "spend")
DATE_HINTS = ("date", "time", "day", "month", "year", "timestamp", "created", "updated")


# --------------------------------------------------------------------- #
def parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse to datetimes without blowing up on mixed timezones.

    pandas raises rather than guessing when a column mixes offsets. That is the
    right default, but profiling must survive dirty data - so the mixed case is
    normalised to UTC and reported separately as a timezone_inconsistency issue.
    """
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
        except (ValueError, TypeError):
            return pd.to_datetime(series, errors="coerce", utc=True)


def infer_type(series: pd.Series, name: str) -> str:
    """numeric | categorical | datetime | text | id | boolean"""
    lower = name.lower()
    non_null = series.dropna()

    if len(non_null) == 0:
        return "text"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        # a numeric column that is unique per row and named like a key is an id
        if any(h in lower for h in ID_HINTS) and non_null.nunique() > 0.9 * len(non_null):
            return "id"
        return "numeric"

    # object dtype - try dates before giving up.
    # The threshold is deliberately low: a date column where a quarter of the
    # values are unparseable is still a date column with a quality problem, and
    # classifying it as categorical would hide exactly the issue worth reporting.
    if any(h in lower for h in DATE_HINTS):
        parsed = parse_datetimes(non_null.head(200))
        if parsed.notna().mean() > 0.6:
            return "datetime"

    ratio = non_null.nunique() / len(non_null)
    if any(h in lower for h in ID_HINTS) and ratio > 0.9:
        return "id"
    if ratio < 0.5 and non_null.nunique() <= 50:
        return "categorical"
    avg_len = non_null.astype(str).str.len().mean()
    return "text" if avg_len > 50 else "categorical"


def semantic_label(name: str, inferred: str) -> str | None:
    """Heuristic placeholder. The LLM Profiler Agent refines this in Phase 7."""
    lower = name.lower()
    if inferred == "datetime":
        return "date"
    if inferred == "id":
        return "identifier"
    if any(h in lower for h in ("region", "country", "city", "state", "zone", "territory")):
        return "geography"
    if any(h in lower for h in ("product", "sku", "item", "category")):
        return "product"
    if any(h in lower for h in ("customer", "client", "user", "account")):
        return "customer"
    if any(h in lower for h in AMOUNT_HINTS):
        return "measure"
    return None


def column_statistics(series: pd.Series, inferred: str) -> dict:
    non_null = series.dropna()
    stats: dict = {"count": int(len(non_null))}
    if len(non_null) == 0:
        return stats

    if inferred == "numeric":
        stats.update(
            min=float(non_null.min()),
            max=float(non_null.max()),
            mean=float(non_null.mean()),
            median=float(non_null.median()),
            std=float(non_null.std()) if len(non_null) > 1 else 0.0,
            q1=float(non_null.quantile(0.25)),
            q3=float(non_null.quantile(0.75)),
        )
    elif inferred == "datetime":
        parsed = parse_datetimes(non_null).dropna()
        if len(parsed):
            stats.update(min=str(parsed.min()), max=str(parsed.max()))
    else:
        top = non_null.astype(str).value_counts().head(10)
        stats["top_values"] = {str(k): int(v) for k, v in top.items()}
    return stats


# --------------------------------------------------------------------- #
# Quality checks (proposal Sec.10 table)
# --------------------------------------------------------------------- #
def column_quality_issues(series: pd.Series, name: str, inferred: str) -> list[dict]:
    issues: list[dict] = []
    total = len(series)
    non_null = series.dropna()
    null_pct = (total - len(non_null)) / total * 100 if total else 0.0

    if null_pct > 0:
        severity = "high" if null_pct > 40 else "medium" if null_pct > 10 else "low"
        issues.append(
            {
                "issue": "missing_values",
                "severity": severity,
                "detail": f"{null_pct:.1f}% of values are missing",
                "affected_rows": int(total - len(non_null)),
            }
        )

    if len(non_null) and non_null.nunique() == 1:
        issues.append(
            {
                "issue": "constant_column",
                "severity": "low",
                "detail": "every non-null value is identical - carries no information",
                "affected_rows": int(len(non_null)),
            }
        )

    if inferred == "numeric" and len(non_null) > 4:
        # Proposal Sec.10 asks for IQR and z-score/robust methods. Both run: IQR
        # is reported, and the robust modified z-score (MAD-based) is reported
        # separately because it catches heavy-tailed cases IQR misses and is not
        # itself distorted by the outliers it is looking for.
        q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = non_null[(non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)]
            if len(outliers):
                issues.append(
                    {
                        "issue": "outliers",
                        "severity": "medium" if len(outliers) / len(non_null) > 0.05 else "low",
                        "detail": f"{len(outliers)} values outside 1.5*IQR",
                        "affected_rows": int(len(outliers)),
                        "method": "iqr",
                    }
                )

        median = non_null.median()
        deviations = (non_null - median).abs()
        mad = float(deviations.median())
        # When most values are identical the MAD is zero and the usual formula
        # divides by nothing - exactly the case where a single wild value is
        # most obvious. Fall back to the mean absolute deviation there.
        if mad > 0:
            scale, constant = mad, 0.6745
        else:
            scale, constant = float(deviations.mean()), 0.7979
        if scale > 0:
            modified_z = constant * deviations / scale
            extreme = non_null[modified_z > 3.5]
            if len(extreme):
                issues.append(
                    {
                        "issue": "extreme_outliers",
                        "severity": "medium" if len(extreme) / len(non_null) > 0.02 else "low",
                        "detail": (
                            f"{len(extreme)} values beyond a modified z-score of 3.5 "
                            f"(median {median:.4g}, scale {scale:.4g})"
                        ),
                        "affected_rows": int(len(extreme)),
                        "method": "robust_zscore",
                    }
                )

        if any(h in name.lower() for h in AMOUNT_HINTS):
            negatives = non_null[non_null < 0]
            if len(negatives):
                issues.append(
                    {
                        "issue": "invalid_range",
                        "severity": "high",
                        "detail": f"{len(negatives)} negative values in a column that should not be negative",
                        "affected_rows": int(len(negatives)),
                    }
                )

    if inferred == "categorical" and len(non_null):
        as_str = non_null.astype(str)
        collapsed = as_str.str.strip().str.lower()
        if collapsed.nunique() < as_str.nunique():
            issues.append(
                {
                    "issue": "inconsistent_categories",
                    "severity": "medium",
                    "detail": (
                        f"{as_str.nunique()} distinct labels collapse to "
                        f"{collapsed.nunique()} after normalising case and spacing"
                    ),
                    "affected_rows": int(len(as_str)),
                }
            )

        # Proposal Sec.10 also asks for fuzzy similarity suggestions: labels that
        # survive normalisation but are probably the same thing typed twice
        # ("Norht" / "North"). Reported as a SUGGESTION only - merging on
        # similarity alone would silently destroy real distinctions.
        near = near_duplicate_categories(collapsed)
        if near:
            issues.append(
                {
                    "issue": "similar_categories",
                    "severity": "low",
                    "detail": (
                        f"{len(near)} label pair(s) look like variants of each other: "
                        + "; ".join(f"'{a}' ~ '{b}'" for a, b in near[:5])
                        + ". Review before merging - the system will not merge these."
                    ),
                    "affected_rows": int(len(as_str)),
                    "suggestions": [list(pair) for pair in near],
                }
            )

    # Proposal Sec.10 - inconsistent units.
    if inferred in {"categorical", "text"} and len(non_null):
        units = mixed_units(non_null.astype(str))
        if len(units) > 1:
            issues.append(
                {
                    "issue": "inconsistent_units",
                    "severity": "high",
                    "detail": (
                        f"values carry more than one unit ({', '.join(sorted(units))}), "
                        "so this column cannot be aggregated until converted"
                    ),
                    "affected_rows": int(len(non_null)),
                    "units_found": sorted(units),
                }
            )

    if inferred == "datetime" and len(non_null):
        parsed = parse_datetimes(non_null)
        bad = int(parsed.isna().sum())
        if bad:
            issues.append(
                {
                    "issue": "date_problems",
                    "severity": "high",
                    "detail": f"{bad} values could not be parsed as dates",
                    "affected_rows": bad,
                }
            )
        valid = parsed.dropna()
        if len(valid):
            future = int((valid > pd.Timestamp.now(tz=valid.dt.tz) + pd.Timedelta(days=365)).sum())
            ancient = int((valid < pd.Timestamp("1900-01-01", tz=valid.dt.tz)).sum())
            if future or ancient:
                issues.append(
                    {
                        "issue": "implausible_dates",
                        "severity": "medium",
                        "detail": f"{future} far-future and {ancient} pre-1900 dates",
                        "affected_rows": future + ancient,
                    }
                )

            # Sec.10 also asks for timezone and ordering checks.
            raw = non_null.astype(str)
            offsets = raw.str.extract(r"([+-]\d{2}:?\d{2}|Z)$", expand=False).dropna()
            distinct_offsets = set(offsets.str.replace(":", "", regex=False).unique())
            naive_present = len(offsets) < len(raw)
            if len(distinct_offsets) > 1 or (distinct_offsets and naive_present):
                issues.append(
                    {
                        "issue": "timezone_inconsistency",
                        "severity": "high",
                        "detail": (
                            "values mix timezone offsets"
                            + (f" ({', '.join(sorted(distinct_offsets))})"
                               if distinct_offsets else "")
                            + (" and some carry no offset at all" if naive_present else "")
                            + " - comparisons across them are not reliable"
                        ),
                        "affected_rows": int(len(raw)),
                        "offsets_found": sorted(distinct_offsets),
                    }
                )

            # A date column that is stored out of order is not itself an error,
            # but it breaks any analysis that assumes row order carries time.
            if len(valid) > 2 and not valid.is_monotonic_increasing:
                out_of_order = int((valid.diff().dropna() < pd.Timedelta(0)).sum())
                if out_of_order:
                    issues.append(
                        {
                            "issue": "date_ordering",
                            "severity": "low",
                            "detail": (
                                f"{out_of_order} row(s) go backwards in time; rows are "
                                "not stored in date order, so sort before any "
                                "sequence-dependent analysis"
                            ),
                            "affected_rows": out_of_order,
                        }
                    )
    return issues


def near_duplicate_categories(values: pd.Series, cutoff: float = 0.8,
                              max_labels: int = 60) -> list[tuple[str, str]]:
    """Label pairs that are similar enough to be probable typos of each other."""
    import difflib

    labels = [str(v) for v in values.dropna().unique()[:max_labels]]
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            # short labels are excluded: 'AB' and 'AC' score highly on similarity
            # while being genuinely different codes
            if a == b or abs(len(a) - len(b)) > 3 or min(len(a), len(b)) < 4:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= cutoff:
                pairs.append((a, b))
    return pairs


def mixed_units(values: pd.Series) -> set[str]:
    """Unit tokens attached to otherwise-numeric values, e.g. '5kg' and '11 lb'."""
    import re

    pattern = re.compile(r"^\s*[-+]?\d[\d,._]*\s*([a-zA-Z%$€£]{1,5})\s*$")
    found: set[str] = set()
    for value in values.dropna().head(500):
        match = pattern.match(str(value))
        if match:
            token = match.group(1).lower()
            if token in UNIT_TOKENS:
                found.add(token)
    return found


def dataset_quality_issues(df: pd.DataFrame) -> list[dict]:
    """Whole-table checks: exact and key-based duplicates (proposal Sec.10)."""
    issues = []
    dupes = int(df.duplicated().sum())
    if dupes:
        issues.append(
            {
                "issue": "duplicate_rows",
                "severity": "high" if dupes / len(df) > 0.05 else "medium",
                "detail": f"{dupes} exact duplicate rows",
                "affected_rows": dupes,
            }
        )

    # Isolation-based detection (proposal Sec.10). Applied to whole ROWS across
    # all numeric columns at once, which is what the method is for: a row whose
    # values are each individually plausible but implausible in combination is
    # invisible to the per-column IQR and z-score checks.
    #
    # The threshold is on the anomaly score distribution rather than on
    # `contamination`, because contamination forces a fixed share of the data to
    # be labelled anomalous - so a perfectly clean table would still get flags.
    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) >= 2 and len(df) >= 50:
        try:
            from sklearn.ensemble import IsolationForest

            matrix = df[numeric_columns].dropna()
            if len(matrix) >= 50:
                forest = IsolationForest(random_state=0, n_estimators=150).fit(matrix)
                scores = forest.score_samples(matrix)
                # A robust threshold on the score distribution itself. Mean and
                # standard deviation are pulled by the very anomalies being
                # looked for, so median and MAD are used instead - the same
                # rule already applied to per-column outliers.
                median = float(np.median(scores))
                mad = float(np.median(np.abs(scores - median))) or 1e-9
                modified_z = 0.6745 * (median - scores) / mad
                # Calibrated against clean synthetic tables, where the highest
                # score reached about 5. A threshold of 8 leaves clear headroom,
                # so this fires on genuinely odd rows rather than on the tail of
                # a normal distribution. It is a flag for review, not a defect.
                anomalous = int((modified_z > 8).sum())
                if anomalous:
                    issues.append(
                        {
                            "issue": "anomalous_rows",
                            "severity": "low",
                            "detail": (
                                f"{anomalous} row(s) are unusual across "
                                f"{len(numeric_columns)} numeric columns taken "
                                "together, even though their individual values may "
                                "look normal"
                            ),
                            "affected_rows": anomalous,
                            "method": "isolation_forest",
                            "columns_considered": [str(c) for c in numeric_columns],
                        }
                    )
        except Exception:  # noqa: BLE001 - profiling must never fail on this
            pass

    # Key-based duplicates are the more dangerous kind: the rows differ, so an
    # exact-duplicate check passes, yet the identifier that is supposed to be
    # unique is not - which silently double-counts in every join and aggregate.
    for name in df.columns:
        series = df[name]
        lower = str(name).lower()
        non_null = series.dropna()
        if len(non_null) < 2:
            continue
        # Classified by NAME plus reasonable distinctness, not by infer_type:
        # a key column full of duplicates is no longer unique enough to be
        # inferred as an id, which is the very case this check exists for.
        looks_like_key = (
            any(h in lower for h in ID_HINTS)
            and non_null.nunique() / len(non_null) > 0.3
        )
        if not (looks_like_key or infer_type(series, str(name)) == "id"):
            continue
        repeated = int(len(non_null) - non_null.nunique())
        if repeated:
            offenders = non_null.astype(str).value_counts()
            offenders = offenders[offenders > 1]
            issues.append(
                {
                    "issue": "duplicate_keys",
                    "severity": "high",
                    "detail": (
                        f"'{name}' looks like an identifier but {repeated} value(s) "
                        f"repeat across {len(offenders)} key(s), e.g. "
                        f"{list(offenders.index[:3])}"
                    ),
                    "affected_rows": repeated,
                    "column": str(name),
                }
            )
    return issues


# --------------------------------------------------------------------- #
def profile_version(db: Session, version: DatasetVersion, df: pd.DataFrame,
                    sensitive: list[str] | None = None) -> dict:
    """Profiles a dataset version, persists per-column rows, returns the summary.

    `sensitive` names columns whose sample values must not be stored in the
    profile (proposal Sec.19). They are still profiled - counts, null rates and
    ranges are safe - but their distinct values are withheld.
    """
    from app.models import Dataset

    if sensitive is None:
        dataset = db.get(Dataset, version.dataset_id)
        sensitive = list((dataset.sensitive_columns or {}).get("columns", [])) if dataset else []
    sensitive_set = set(sensitive)

    db.query(DatasetColumn).filter(DatasetColumn.version_id == version.id).delete()

    columns_out = []
    for position, name in enumerate(df.columns):
        series = df[name]
        inferred = infer_type(series, str(name))
        is_sensitive = str(name) in sensitive_set
        issues = column_quality_issues(series, str(name), inferred)
        stats = column_statistics(series, inferred)
        if is_sensitive:
            stats.pop("top_values", None)
            stats["values_withheld"] = "column marked sensitive"
        null_count = int(series.isna().sum())

        db.add(
            DatasetColumn(
                version_id=version.id,
                name=str(name),
                position=position,
                raw_dtype=str(series.dtype),
                inferred_type=inferred,
                semantic_label=semantic_label(str(name), inferred),
                null_count=null_count,
                null_percentage=round(null_count / len(df) * 100, 2) if len(df) else 0.0,
                unique_count=int(series.nunique(dropna=True)),
                statistics=stats,
                quality_issues={"issues": issues} if issues else None,
                sensitive=is_sensitive,
            )
        )
        columns_out.append(
            {
                "name": str(name),
                "inferred_type": inferred,
                "semantic_label": semantic_label(str(name), inferred),
                "null_percentage": round(null_count / len(df) * 100, 2) if len(df) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "statistics": stats,
                "issues": issues,
                "sensitive": is_sensitive,
            }
        )

    ds_issues = dataset_quality_issues(df)
    all_issues = ds_issues + [i for c in columns_out for i in c["issues"]]
    health = _health_score(all_issues)

    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "health_score": health,
        "dataset_issues": ds_issues,
        "issue_counts": {
            "high": sum(1 for i in all_issues if i["severity"] == "high"),
            "medium": sum(1 for i in all_issues if i["severity"] == "medium"),
            "low": sum(1 for i in all_issues if i["severity"] == "low"),
        },
        "columns": columns_out,
    }
    version.profile_summary = summary
    db.flush()
    return summary


def _health_score(issues: list[dict]) -> int:
    """100 = clean. Weighted penalty so one high-severity issue outweighs many low ones."""
    weights = {"high": 12, "medium": 5, "low": 2}
    penalty = sum(weights.get(i["severity"], 2) for i in issues)
    return int(max(0, 100 - penalty))
