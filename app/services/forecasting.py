"""Phase 9 - Forecasting (proposal Sec.12).

Three rules from the proposal are enforced in code, not just documented:
  1. a forecast is never a single number - always point + lower + upper
  2. nothing is shown until it has been backtested against held-out periods
  3. if the series is too short or the backtest is too poor, the forecast is
     withheld or marked low_confidence rather than displayed as fact

The minimum-length guards exist because ARIMA and seasonal models will happily
fit 8 monthly points and return confident nonsense.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.services.profiling import parse_datetimes

from app.config import settings

warnings.filterwarnings("ignore")


def build_series(df: pd.DataFrame, date_column: str, metric: str,
                 freq: str = "M", agg: str = "sum") -> pd.Series:
    from app.services.analytics import normalize_freq

    tmp = df[[date_column, metric]].copy()
    tmp[date_column] = parse_datetimes(tmp[date_column])
    tmp = tmp.dropna(subset=[date_column])
    series = tmp.set_index(date_column)[metric].resample(normalize_freq(freq)).agg(agg)
    return series.dropna()


def prophet_available() -> bool:
    """Prophet is optional: its build is heavy and unavailable on some Pythons."""
    try:
        import prophet  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def has_trend_change(values: np.ndarray, min_segments: int = 3) -> bool:
    """Crude changepoint check: did the slope reverse partway through?

    Proposal Sec.12 reserves Prophet for "series with holidays/trend changes",
    so it is only a candidate when there is an actual trend break to model.
    """
    n = len(values)
    if n < min_segments * 4:
        return False
    thirds = np.array_split(values, 3)
    slopes = []
    for part in thirds:
        if len(part) < 2:
            return False
        slopes.append(float(np.polyfit(np.arange(len(part)), part, 1)[0]))
    scale = float(np.mean(np.abs(values))) or 1.0
    significant = [s for s in slopes if abs(s) > 0.01 * scale]
    if len(significant) < 2:
        return False
    return any(a * b < 0 for a, b in zip(significant, significant[1:]))


def select_model(n: int, seasonal_hint: bool, trend_change: bool = False,
                 has_external_features: bool = False) -> str:
    """Model choice is driven by what the data actually supports (Sec.12).

    Order of preference, each gated on a real condition rather than on novelty:
      insufficient  - too few periods to validate anything
      ets           - short, low-volume series
      gbr           - relevant external features exist (price, stock, promotions)
      prophet       - a trend change is present and prophet is installed
      sarima/arima  - seasonal or plain autoregressive structure

    Whatever is chosen still has to beat a naive baseline in the backtest before
    it is shown, so a wrong choice here degrades to low_confidence rather than
    to a confident wrong answer.
    """
    if n < settings.MIN_SERIES_LENGTH:
        return "insufficient"
    if n < settings.MIN_SERIES_FOR_SEASONAL:
        return "gbr" if has_external_features and n >= 16 else "ets"
    if has_external_features:
        return "gbr"
    if trend_change and prophet_available():
        return "prophet"
    return "sarima" if seasonal_hint else "arima"


# --------------------------------------------------------------------- #
def _fit_predict(values: np.ndarray, model: str, horizon: int, period: int = 12,
                 exog: np.ndarray | None = None):
    """Returns (point_forecasts, lower, upper)."""
    if model == "naive":
        last = float(values[-1])
        resid_std = float(np.std(np.diff(values))) if len(values) > 2 else 0.0
        points = np.full(horizon, last)
        margin = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
        return points, points - margin, points + margin

    if model == "moving_average":
        window = min(3, len(values))
        last = float(np.mean(values[-window:]))
        resid_std = float(np.std(values[-window:])) if window > 1 else 0.0
        points = np.full(horizon, last)
        margin = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
        return points, points - margin, points + margin

    if model == "ets":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(values, trend="add", seasonal=None,
                                       initialization_method="estimated").fit()
        points = np.asarray(fit.forecast(horizon), dtype=float)
        resid_std = float(np.std(fit.resid)) if hasattr(fit, "resid") else 0.0
        margin = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
        return points, points - margin, points + margin

    if model in {"arima", "sarima"}:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, period) if model == "sarima" else (0, 0, 0, 0)
        # statsmodels warns loudly on short or hard-to-converge series. That is
        # already handled by the backtest gate below, which marks such a model
        # low_confidence, so the warning is noise rather than information.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(values, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
        res = fit.get_forecast(steps=horizon)
        ci = res.conf_int(alpha=0.05)
        return (np.asarray(res.predicted_mean, dtype=float),
                np.asarray(ci[:, 0], dtype=float),
                np.asarray(ci[:, 1], dtype=float))

    if model == "prophet":
        try:
            from prophet import Prophet
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "prophet is not installed. Install it with `pip install prophet`, "
                "or leave it out - the selector falls back to SARIMA."
            ) from exc

        index = pd.date_range("2000-01-01", periods=len(values), freq="MS")
        frame = pd.DataFrame({"ds": index, "y": values})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_fit = Prophet(interval_width=0.95, daily_seasonality=False,
                                weekly_seasonality=False,
                                yearly_seasonality=len(values) >= 24)
            model_fit.fit(frame)
            future = model_fit.make_future_dataframe(periods=horizon, freq="MS")
            forecast = model_fit.predict(future).tail(horizon)
        return (forecast["yhat"].to_numpy(dtype=float),
                forecast["yhat_lower"].to_numpy(dtype=float),
                forecast["yhat_upper"].to_numpy(dtype=float))

    if model == "gbr":
        return _gbr_predict(values, horizon, exog)

    raise ValueError(f"Unknown model: {model}")


def _lag_features(values: np.ndarray, exog: np.ndarray | None, lags: int = 3):
    """Build lag-1..lag-n plus a rolling mean, and align any external features."""
    rows, targets = [], []
    for i in range(lags, len(values)):
        window = values[i - lags:i]
        row = list(window) + [float(np.mean(window))]
        if exog is not None:
            row.extend(exog[i])
        rows.append(row)
        targets.append(values[i])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _gbr_predict(values: np.ndarray, horizon: int, exog: np.ndarray | None,
                 lags: int = 3):
    """Gradient-boosted regressor on lag features (proposal Sec.12).

    Forecasts recursively: each predicted step becomes the input for the next.
    External features are held at their last observed value, which is the
    honest default - the model is not given the future it is trying to predict.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    if len(values) < lags + 6:
        raise ValueError("Not enough observations for lag features")

    X, y = _lag_features(values, exog, lags)
    model = GradientBoostingRegressor(random_state=0, n_estimators=200,
                                      max_depth=3, learning_rate=0.05)

    # Uncertainty is measured on data the model did not train on. In-sample
    # residuals from a boosted tree are near zero, which produced confidence
    # bands so narrow that they never contained the truth - a range that is
    # always wrong is worse than no range at all.
    holdout = max(3, len(y) // 5)
    if len(y) > holdout + 6:
        model.fit(X[:-holdout], y[:-holdout])
        residual_std = float(np.std(y[-holdout:] - model.predict(X[-holdout:])))
    else:
        residual_std = float(np.std(y - model.predict(X)))
    model.fit(X, y)
    residual_std = residual_std or 0.0
    last_exog = exog[-1] if exog is not None else None

    window = list(values[-lags:])
    points = []
    for _ in range(horizon):
        row = window + [float(np.mean(window))]
        if last_exog is not None:
            row.extend(last_exog)
        step = float(model.predict(np.asarray([row], dtype=float))[0])
        points.append(step)
        window = window[1:] + [step]

    points = np.asarray(points, dtype=float)
    # uncertainty widens with the horizon because each step compounds the last
    margin = 1.96 * residual_std * np.sqrt(np.arange(1, horizon + 1))
    return points, points - margin, points + margin


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def backtest(values: np.ndarray, model: str, horizon: int = 1, folds: int = 3,
             period: int = 12, exog: np.ndarray | None = None) -> dict:
    """Rolling-origin cross-validation (proposal Sec.12 'validation before trust')."""
    errors_model, errors_naive = [], []
    min_train = max(settings.MIN_SERIES_LENGTH - 2, 6)

    for i in range(folds, 0, -1):
        split = len(values) - i * horizon
        if split < min_train:
            continue
        train, test = values[:split], values[split : split + horizon]
        if len(test) == 0:
            continue
        train_exog = exog[:split] if exog is not None else None
        try:
            pred, _, _ = _fit_predict(train, model, len(test), period, train_exog)
            naive, _, _ = _fit_predict(train, "naive", len(test), period)
        except Exception:  # noqa: BLE001
            continue
        errors_model.append(_mape(test, pred))
        errors_naive.append(_mape(test, naive))

    if not errors_model:
        return {"folds": 0, "mape": None, "naive_mape": None, "beats_naive": None}

    mape = float(np.mean(errors_model))
    naive_mape = float(np.mean(errors_naive))
    return {
        "folds": len(errors_model),
        "mape": round(mape, 2),
        "naive_mape": round(naive_mape, 2),
        "beats_naive": bool(mape < naive_mape),
        "rmse_note": "MAPE reported; RMSE available per-fold if required.",
    }


# --------------------------------------------------------------------- #
def forecast_series(series: pd.Series, horizon: int = 3, freq: str = "M",
                    seasonal_hint: bool | None = None,
                    exog: pd.DataFrame | None = None) -> dict:
    values = series.astype(float).values
    n = len(values)
    exog_values = None
    exog_names: list[str] = []
    if exog is not None and len(exog) == n and not exog.empty:
        exog_values = exog.to_numpy(dtype=float)
        exog_names = [str(c) for c in exog.columns]

    if n < settings.MIN_SERIES_LENGTH:
        return {
            "reliability": "withheld_insufficient_data",
            "withheld_reason": (
                f"Only {n} periods available; at least {settings.MIN_SERIES_LENGTH} are "
                "required before a forecast can be validated. Showing one anyway would "
                "present an unvalidated guess as evidence."
            ),
            "series_length": n,
            "model_name": "none",
            "predictions": [],
            "backtest_metrics": None,
        }

    period = {"M": 12, "ME": 12, "MS": 12, "W": 52, "D": 7, "Q": 4, "QE": 4}.get(freq, 12)
    if seasonal_hint is None:
        seasonal_hint = n >= 2 * period

    model = select_model(
        n, bool(seasonal_hint),
        trend_change=has_trend_change(values),
        has_external_features=exog_values is not None,
    )
    attempted = [model]
    try:
        points, lower, upper = _fit_predict(values, model, horizon, period, exog_values)
    except Exception:  # noqa: BLE001
        model = "ets"
        attempted.append(model)
        try:
            points, lower, upper = _fit_predict(values, model, horizon, period)
        except Exception:  # noqa: BLE001
            model = "moving_average"
            attempted.append(model)
            points, lower, upper = _fit_predict(values, model, horizon, period)

    metrics = backtest(values, model, horizon=1, folds=3, period=period,
                       exog=exog_values if model == "gbr" else None)

    reliability = "ok"
    withheld_reason = None
    if metrics["mape"] is None:
        reliability = "low_confidence"
        withheld_reason = "Not enough history to backtest; treat this projection as indicative only."
    elif metrics["mape"] > settings.MAX_ACCEPTABLE_MAPE:
        reliability = "low_confidence"
        withheld_reason = (
            f"Backtest error is {metrics['mape']:.1f}% MAPE, above the "
            f"{settings.MAX_ACCEPTABLE_MAPE}% threshold."
        )
    elif metrics.get("beats_naive") is False:
        reliability = "low_confidence"
        withheld_reason = (
            "The model did not outperform a naive last-value baseline on held-out periods."
        )

    last_index = series.index[-1]
    from app.services.analytics import normalize_freq
    offset = pd.tseries.frequencies.to_offset(normalize_freq(freq))
    predictions = []
    for i in range(horizon):
        period_label = str((last_index + offset * (i + 1)).date())
        predictions.append(
            {
                "period": period_label,
                "point": round(float(points[i]), 2),
                "lower": round(float(lower[i]), 2),
                "upper": round(float(upper[i]), 2),
            }
        )

    return {
        "reliability": reliability,
        "withheld_reason": withheld_reason,
        "series_length": n,
        "model_name": model,
        "frequency": freq,
        "horizon_periods": horizon,
        "predictions": predictions,
        "backtest_metrics": metrics,
        "models_attempted": attempted,
        "external_features": exog_names,
        "prophet_available": prophet_available(),
        "history": [round(float(v), 2) for v in values],
        # the observed periods, so a chart can label the past with real dates
        # instead of a relative counter that clashes with the forecast labels
        "history_periods": [str(pd.Timestamp(i).date()) for i in series.index],
        "framing": "Projection assumes current patterns continue; it is not a guarantee.",
    }