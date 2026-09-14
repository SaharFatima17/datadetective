"""Phase 5 - Analytics tools (proposal Sec.9).

Every number that reaches a finding is produced here. The LLM chooses WHICH of
these to call and with what arguments; it never computes the result itself.

Note there is no `exec()` anywhere. `run_dataframe_code` in the proposal is
implemented as a whitelist of named operations - an agent cannot ask for
arbitrary Python, which removes the sandbox-escape problem entirely (Sec.19).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.profiling import parse_datetimes

from app.config import settings

ALLOWED_OPS = {"describe", "value_counts", "groupby_aggregate", "correlation",
               "time_series", "period_contribution", "interaction_contribution"}
ALLOWED_AGGS = {"sum", "mean", "median", "count", "min", "max", "std", "nunique"}

# pandas 2.2 renamed the month/quarter/year offsets; accept the old spellings
FREQ_ALIASES = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}


def normalize_freq(freq: str) -> str:
    return FREQ_ALIASES.get(freq, freq)


# --------------------------------------------------------------------- #
def run_dataframe_op(df: pd.DataFrame, operation: str, params: dict) -> dict:
    if operation not in ALLOWED_OPS:
        raise ValueError(f"Operation not permitted. Allowed: {sorted(ALLOWED_OPS)}")

    if operation == "describe":
        cols = params.get("columns") or df.select_dtypes(include=np.number).columns.tolist()
        return {"operation": "describe", "result": df[cols].describe().to_dict()}

    if operation == "value_counts":
        col = _require(params, "column")
        limit = int(params.get("limit", 20))
        vc = df[col].value_counts().head(limit)
        return {
            "operation": "value_counts",
            "column": col,
            "result": {str(k): int(v) for k, v in vc.items()},
        }

    if operation == "groupby_aggregate":
        by = _require(params, "by")
        by = [by] if isinstance(by, str) else by
        metric = _require(params, "metric")
        agg = params.get("agg", "sum")
        if agg not in ALLOWED_AGGS:
            raise ValueError(f"Aggregation not permitted. Allowed: {sorted(ALLOWED_AGGS)}")
        grouped = df.groupby(by, dropna=False)[metric].agg(agg).sort_values(ascending=False)
        total = float(grouped.sum()) if agg == "sum" else None
        rows = []
        for key, value in grouped.items():
            label = " | ".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
            row = {"group": label, "value": float(value)}
            if total:
                row["share_pct"] = round(float(value) / total * 100, 2)
            rows.append(row)
        return {"operation": "groupby_aggregate", "by": by, "metric": metric,
                "agg": agg, "total": total, "result": rows}

    if operation == "correlation":
        cols = params.get("columns") or df.select_dtypes(include=np.number).columns.tolist()
        corr = df[cols].corr(numeric_only=True)
        pairs = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1 :]:
                v = corr.loc[a, b]
                if pd.notna(v):
                    pairs.append({"a": a, "b": b, "correlation": round(float(v), 4)})
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        return {"operation": "correlation", "result": pairs,
                "note": "Correlation is not causation (proposal Sec.19)."}

    if operation == "time_series":
        date_col = _require(params, "date_column")
        metric = _require(params, "metric")
        freq = normalize_freq(params.get("freq", "M"))
        agg = params.get("agg", "sum")
        tmp = df[[date_col, metric]].copy()
        tmp[date_col] = parse_datetimes(tmp[date_col])
        tmp = tmp.dropna(subset=[date_col])
        series = tmp.set_index(date_col)[metric].resample(freq).agg(agg)
        return {
            "operation": "time_series",
            "date_column": date_col,
            "metric": metric,
            "freq": freq,
            "result": [
                {"period": str(idx.date()), "value": (None if pd.isna(v) else float(v))}
                for idx, v in series.items()
            ],
        }

    if operation == "period_contribution":
        # Proposal Sec.11 "contribution analysis by region" and the Sec.18 example
        # ("Region B contributed 71.6% of the total decline").
        #
        # This answers a different question from groupby_aggregate: not "which group
        # is biggest" but "which group drove the CHANGE between two periods". A group
        # can be small overall and still cause the entire decline.
        date_col = _require(params, "date_column")
        metric = _require(params, "metric")
        by = _require(params, "by")
        split = params.get("split")

        tmp = df[[date_col, metric, by]].copy()
        tmp[date_col] = parse_datetimes(tmp[date_col])
        tmp = tmp.dropna(subset=[date_col])
        if tmp.empty:
            raise ValueError("No parseable dates in the date column")

        cutoff = pd.Timestamp(split) if split else tmp[date_col].quantile(0.67)
        previous = tmp[tmp[date_col] < cutoff]
        current = tmp[tmp[date_col] >= cutoff]
        if previous.empty or current.empty:
            raise ValueError("The split produced an empty period")

        # normalise for unequal period lengths - otherwise a shorter current
        # period looks like a decline in every group
        prev_periods = max(previous[date_col].dt.to_period("M").nunique(), 1)
        curr_periods = max(current[date_col].dt.to_period("M").nunique(), 1)

        prev = previous.groupby(by)[metric].sum() / prev_periods
        curr = current.groupby(by)[metric].sum() / curr_periods
        groups = sorted(set(prev.index) | set(curr.index), key=str)

        changes = {g: float(curr.get(g, 0.0) - prev.get(g, 0.0)) for g in groups}
        total_change = sum(changes.values())
        direction = "decline" if total_change < 0 else "increase"

        rows = []
        for g in groups:
            change = changes[g]
            # share of the movement, signed the same way as the total
            share = (change / total_change * 100) if total_change else 0.0
            rows.append({
                "group": str(g),
                "previous_per_period": round(float(prev.get(g, 0.0)), 2),
                "current_per_period": round(float(curr.get(g, 0.0)), 2),
                "change": round(change, 2),
                "change_pct": round(change / prev.get(g, 0.0) * 100, 2) if prev.get(g, 0.0) else None,
                "contribution_to_total_change_pct": round(share, 2),
            })
        rows.sort(key=lambda r: r["contribution_to_total_change_pct"], reverse=True)

        seasonality = _seasonal_check(tmp, date_col, metric, by, cutoff,
                                      rows[0]["group"] if rows else None)
        for row in rows:
            row["contribution_ci"] = _contribution_interval(
                tmp, date_col, metric, by, cutoff, row["group"])

        return {
            "operation": "period_contribution",
            "seasonality": seasonality,
            "by": by,
            "metric": metric,
            "split_date": str(cutoff.date()),
            "direction": direction,
            "total_change_per_period": round(total_change, 2),
            "total_change_pct": round(total_change / prev.sum() * 100, 2) if prev.sum() else None,
            "result": rows,
            "note": (
                "Values are normalised per period so unequal period lengths do not "
                "look like a change. Contribution shows concentration, not cause."
            ),
        }

    if operation == "interaction_contribution":
        # Refining an answer, not competing with it.
        #
        # Testing one column at a time gives "the decline is in the South". That
        # is true and often where the investigation stops — but "the South" may
        # be four products of which one collapsed and three are fine. Acting on
        # "the South" then spends effort on three healthy products.
        #
        # So this takes the group a single-column analysis already identified
        # and asks whether, inside it, the movement sits in one value of a
        # second column. It reports only when it does; otherwise the simpler
        # statement is the honest one.
        date_col = _require(params, "date_column")
        metric = _require(params, "metric")
        dims = params.get("by") or []
        if len(dims) < 2:
            raise ValueError("interaction_contribution needs two columns in 'by'")
        primary, secondary = dims[0], dims[1]

        tmp = df[[date_col, metric, primary, secondary]].copy()
        tmp[date_col] = parse_datetimes(tmp[date_col])
        tmp = tmp.dropna(subset=[date_col])
        if tmp.empty:
            raise ValueError("No parseable dates in the date column")

        outer = run_dataframe_op(
            tmp, "period_contribution",
            {"date_column": date_col, "metric": metric, "by": primary,
             "split": params.get("split")})
        if not outer["result"]:
            raise ValueError("No groups to analyse")

        lead = outer["result"][0]
        inside = tmp[tmp[primary].astype(str) == str(lead["group"])]
        if inside.empty:
            raise ValueError("Leading group has no rows")

        inner = run_dataframe_op(
            inside, "period_contribution",
            {"date_column": date_col, "metric": metric, "by": secondary,
             "split": outer["split_date"]})

        top_inner = inner["result"][0] if inner["result"] else None
        inner_share = top_inner["contribution_to_total_change_pct"] if top_inner else 0.0
        n_values = len(inner["result"])
        even = 100.0 / n_values if n_values else 100.0

        # Concentrated means one value carries far more than an even split, and
        # there was more than one value to begin with.
        concentrated = bool(top_inner and n_values > 1
                            and inner_share > max(60.0, even * 1.8))

        return {
            "operation": "interaction_contribution",
            "by": [primary, secondary],
            "metric": metric,
            "split_date": outer["split_date"],
            "direction": outer["direction"],
            "leading_group": lead["group"],
            "leading_group_share_pct": lead["contribution_to_total_change_pct"],
            "within_leading_group": inner["result"][:8],
            "refined_to": (f"{lead['group']} × {top_inner['group']}"
                           if concentrated else None),
            "refined_share_within_group_pct": round(float(inner_share), 2),
            "values_examined": n_values,
            "worth_reporting": concentrated,
            "note": (
                f"Of the {outer['direction']} inside {primary} = "
                f"'{lead['group']}', this is how it splits across {secondary}. "
                "Reported only when one value carries most of it; otherwise the "
                "single-column statement is the accurate one."
            ),
        }

    raise ValueError(f"Unhandled operation: {operation}")


def _seasonal_check(tmp, date_col: str, metric: str, by: str,
                    cutoff, top_group) -> dict:
    """Does the change survive a like-for-like comparison with a year earlier?

    Every finding this system produces carries the caveat that seasonality is
    not controlled for, and until now nothing acted on it. A drop that appears
    every December is not a cause, it is a calendar.

    The test is deliberately plain: compare the same months one year apart. It
    needs no model, no assumption about the shape of the season, and it is easy
    to explain — which matters more here than a decomposition nobody can check.
    """
    if top_group is None:
        return {"checked": False, "reason": "no group to test"}

    series = tmp[tmp[by].astype(str) == str(top_group)]
    if series.empty:
        return {"checked": False, "reason": "group not found"}

    monthly = (series.set_index(date_col)[metric]
               .resample("ME").sum().sort_index())
    if len(monthly) < 24:
        return {
            "checked": False,
            "reason": (f"only {len(monthly)} months of data; a year-on-year "
                       "comparison needs at least 24"),
        }

    current = monthly[monthly.index >= cutoff]
    if current.empty:
        return {"checked": False, "reason": "no months after the split"}

    # the same calendar months, one year earlier
    pairs = []
    for period, value in current.items():
        prior = period - pd.DateOffset(years=1)
        match = monthly[(monthly.index.year == prior.year)
                        & (monthly.index.month == prior.month)]
        if not match.empty:
            pairs.append((float(value), float(match.iloc[0])))

    if len(pairs) < 2:
        return {"checked": False,
                "reason": "not enough matching months a year earlier"}

    now = sum(p[0] for p in pairs)
    year_ago = sum(p[1] for p in pairs)
    yoy = ((now - year_ago) / year_ago * 100) if year_ago else None

    return {
        "checked": True,
        "months_compared": len(pairs),
        "same_months_last_year": round(year_ago, 2),
        "same_months_this_year": round(now, 2),
        "year_on_year_pct": round(yoy, 2) if yoy is not None else None,
        "survives": bool(yoy is not None and yoy < -5),
        "note": (
            "Compares the same calendar months a year apart, so a pattern that "
            "repeats every year cancels out. A change that survives this is not "
            "explained by the season alone."
        ),
    }


def _contribution_interval(tmp, date_col: str, metric: str, by: str,
                           cutoff, group, draws: int = 200) -> dict | None:
    """A range around a group's share of the movement.

    The share is computed from a sample of rows, so it is an estimate. Reporting
    94.27% with no interval reads as a measurement rather than an estimate —
    and this project reports a band around its forecasts for exactly the same
    reason. Resampling rows with replacement gives the spread without assuming
    any distribution.
    """
    import numpy as np

    # Real exports carry missing values. A single NaN propagates through the
    # resampled sums and comes out the other end as a NaN interval, which is
    # not valid JSON and takes the whole investigation down with it. Drop them
    # here rather than discovering it at the database.
    clean = tmp.dropna(subset=[metric])
    before = clean[clean[date_col] < cutoff]
    after = clean[clean[date_col] >= cutoff]
    if len(before) < 30 or len(after) < 30:
        return None            # too few rows for the spread to mean anything

    rng = np.random.default_rng(17)          # fixed: the same data gives the same band
    prev_periods = max(before[date_col].dt.to_period("M").nunique(), 1)
    curr_periods = max(after[date_col].dt.to_period("M").nunique(), 1)
    mask_b = before[by].astype(str) == str(group)
    mask_a = after[by].astype(str) == str(group)
    vals_b, vals_a = before[metric].to_numpy(), after[metric].to_numpy()
    grp_b, grp_a = mask_b.to_numpy(), mask_a.to_numpy()

    shares = []
    for _ in range(draws):
        ib = rng.integers(0, len(vals_b), len(vals_b))
        ia = rng.integers(0, len(vals_a), len(vals_a))
        prev_g = vals_b[ib][grp_b[ib]].sum() / prev_periods
        curr_g = vals_a[ia][grp_a[ia]].sum() / curr_periods
        total = vals_a[ia].sum() / curr_periods - vals_b[ib].sum() / prev_periods
        if total and np.isfinite(total):
            shares.append((curr_g - prev_g) / total * 100)

    shares = [x for x in shares if np.isfinite(x)]
    if len(shares) < draws // 2:
        return None
    low, high = np.percentile(shares, [2.5, 97.5])
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return {"low": round(float(low), 1), "high": round(float(high), 1),
            "method": f"bootstrap, {draws} resamples, 95% interval"}


def _require(params: dict, key: str):
    if key not in params:
        raise ValueError(f"Missing required parameter: {key}")
    return params[key]


# --------------------------------------------------------------------- #
def run_readonly_sql(parquet_path: str, query: str) -> dict:
    """DuckDB over the stored parquet file. SELECT only (proposal Sec.19).

    DuckDB opens the file itself, so an encrypted version is decrypted to a
    temporary file that is deleted as soon as the query finishes.
    """
    import duckdb

    from app.core import crypto

    stripped = query.strip().rstrip(";")
    if not stripped.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT queries are permitted")
    for banned in ("insert", "update", "delete", "drop", "alter", "truncate",
                   "create", "attach", "copy", "install", "load"):
        if f" {banned} " in f" {stripped.lower()} ":
            raise ValueError(f"Query contains a forbidden keyword: {banned}")

    with crypto.materialize(parquet_path) as readable:
        con = duckdb.connect(":memory:")
        try:
            # DuckDB does not bind parameters inside CREATE VIEW, so the path is
            # inlined with quote-escaping instead.
            safe_path = str(Path(readable)).replace("'", "''")
            con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{safe_path}')")
            df = con.execute(f"SELECT * FROM ({stripped}) LIMIT {settings.MAX_SQL_ROWS}").df()
        finally:
            con.close()

    return {
        "query": stripped,
        "row_count": len(df),
        "columns": [str(c) for c in df.columns],
        "rows": df.head(500).replace({np.nan: None}).to_dict(orient="records"),
    }


# --------------------------------------------------------------------- #
def statistical_test(df: pd.DataFrame, test: str, params: dict) -> dict:
    """Proposal Sec.11 - tests with effect sizes and stated assumptions."""
    from scipy import stats as st

    if test == "ttest":
        value_col = _require(params, "value_column")
        group_col = _require(params, "group_column")
        groups = params.get("groups")
        series = df[[value_col, group_col]].dropna()
        levels = groups or series[group_col].astype(str).unique().tolist()
        if len(levels) != 2:
            raise ValueError("ttest needs exactly two groups; pass `groups` to choose them")
        a = series[series[group_col].astype(str) == str(levels[0])][value_col]
        b = series[series[group_col].astype(str) == str(levels[1])][value_col]
        if len(a) < 2 or len(b) < 2:
            raise ValueError("each group needs at least 2 observations")
        stat, p = st.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) /
                         (len(a) + len(b) - 2))
        d = float((a.mean() - b.mean()) / pooled) if pooled else 0.0
        return {
            "test": "welch_ttest",
            "groups": [str(levels[0]), str(levels[1])],
            "means": [float(a.mean()), float(b.mean())],
            "n": [int(len(a)), int(len(b))],
            "statistic": float(stat),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "cohens_d": round(d, 4),
            "effect_size_label": _effect_label(abs(d)),
            "assumptions": "Welch's t-test; does not assume equal variances. Assumes independent observations.",
        }

    if test == "chi_square":
        a_col, b_col = _require(params, "column_a"), _require(params, "column_b")
        table = pd.crosstab(df[a_col], df[b_col])
        chi2, p, dof, _ = st.chi2_contingency(table)
        n = table.values.sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if min(table.shape) > 1 else 0.0
        return {
            "test": "chi_square",
            "statistic": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "significant": bool(p < 0.05),
            "cramers_v": round(cramers_v, 4),
            "assumptions": "Expected cell counts should be >= 5; check the contingency table.",
        }

    if test == "correlation":
        a_col, b_col = _require(params, "column_a"), _require(params, "column_b")
        sub = df[[a_col, b_col]].dropna()
        r, p = st.pearsonr(sub[a_col], sub[b_col])
        return {
            "test": "pearson_correlation",
            "r": float(r),
            "r_squared": float(r ** 2),
            "p_value": float(p),
            "n": int(len(sub)),
            "significant": bool(p < 0.05),
            "assumptions": "Assumes a linear relationship. Correlation does not establish causation.",
        }

    if test == "linear_regression":
        y_col = _require(params, "y_column")
        x_cols = _require(params, "x_columns")
        x_cols = [x_cols] if isinstance(x_cols, str) else x_cols
        import statsmodels.api as sm

        sub = df[[y_col] + x_cols].dropna()
        X = sm.add_constant(sub[x_cols])
        model = sm.OLS(sub[y_col], X).fit()
        return {
            "test": "ols_regression",
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "n": int(len(sub)),
            "coefficients": {
                k: {"coef": float(v), "p_value": float(model.pvalues[k])}
                for k, v in model.params.items()
            },
            "assumptions": "OLS assumes linearity, independent errors and constant variance.",
        }

    raise ValueError("Unknown test. Use: ttest, chi_square, correlation, linear_regression")


def _effect_label(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# --------------------------------------------------------------------- #
def render_chart(df: pd.DataFrame, spec: dict, output_dir: Path) -> str:
    """Renders a chart to PNG and returns its path (proposal Sec.9 render_chart)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_type = spec.get("type", "bar")
    title = spec.get("title", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{uuid.uuid4().hex}.png"

    fig, ax = plt.subplots(figsize=(9, 5))

    def _readable_axis(axis) -> None:
        """Print 120,000 rather than 1.2 with a '1e5' hidden in the corner.

        Matplotlib's offset notation is easy to crop out of an embedded image,
        and a chart whose axis silently reads 0.00 to 1.00 when the values are
        hundreds of thousands is worse than no chart.
        """
        from matplotlib.ticker import FuncFormatter

        axis.ticklabel_format(style="plain", axis="y")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    if chart_type == "bar":
        x, y = spec["x"], spec["y"]
        agg = spec.get("agg", "sum")
        data = df.groupby(x)[y].agg(agg).sort_values(ascending=False).head(20)
        ax.bar([str(i) for i in data.index], data.values, color="#0a8f73")
        ax.set_xlabel(x)
        ax.set_ylabel(f"{agg}({y})")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        _readable_axis(ax)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "line":
        x, y = spec["x"], spec["y"]
        tmp = df[[x, y]].copy()
        tmp[x] = parse_datetimes(tmp[x])
        tmp = tmp.dropna().sort_values(x)
        if spec.get("freq"):
            tmp = tmp.set_index(x)[y].resample(spec["freq"]).agg(spec.get("agg", "sum"))
            ax.plot(tmp.index, tmp.values, marker="o", color="#4C72B0")
        else:
            ax.plot(tmp[x], tmp[y], color="#4C72B0")
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    elif chart_type == "hist":
        col = spec["column"]
        ax.hist(df[col].dropna(), bins=spec.get("bins", 30), color="#4C72B0")
        ax.set_xlabel(col)
        ax.set_ylabel("frequency")

    elif chart_type == "scatter":
        x, y = spec["x"], spec["y"]
        sub = df[[x, y]].dropna()
        ax.scatter(sub[x], sub[y], alpha=0.6, color="#4C72B0")
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    elif chart_type == "forecast":
        # historical line plus projected band (proposal Sec.17 forecast panel)
        hist, fc = spec["history"], spec["forecast"]
        ax.plot(range(len(hist)), hist, marker="o", label="actual", color="#4C72B0")
        start = len(hist) - 1
        idx = range(start, start + len(fc) + 1)
        points = [hist[-1]] + [f["point"] for f in fc]
        lower = [hist[-1]] + [f["lower"] for f in fc]
        upper = [hist[-1]] + [f["upper"] for f in fc]
        ax.plot(idx, points, marker="o", linestyle="--", label="forecast", color="#DD8452")
        ax.fill_between(idx, lower, upper, alpha=0.2, color="#DD8452", label="confidence band")
        ax.legend()
    else:
        plt.close(fig)
        raise ValueError(f"Unknown chart type: {chart_type}")

    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)