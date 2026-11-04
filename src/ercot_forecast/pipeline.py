"""Run the day-ahead study and write forecasts, metrics, figures, and a summary."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ercot_forecast.config import (
    FIGURES,
    FOLDS,
    FORECASTS_PATH,
    LOAD_TARGET,
    METRICS_PATH,
    PRICE_TARGET,
    SUMMARY_PATH,
)
from ercot_forecast.download import load_hourly_frame, load_zone_weights
from ercot_forecast.features import build_features
from ercot_forecast.metrics import summarize_forecasts
from ercot_forecast.models import cross_validate
from ercot_forecast.plots import save_all
from ercot_forecast.prices import price_context

LOGGER = logging.getLogger(__name__)


def run_analysis(frame: pd.DataFrame | None = None, save: bool = True) -> dict:
    hourly = frame if frame is not None else load_hourly_frame()
    weights = load_zone_weights() if frame is None else _weights_from_frame(hourly)
    featured = build_features(hourly, weights)
    forecasts, selection = cross_validate(featured, FOLDS)
    metric_tables = summarize_forecasts(forecasts)
    context = price_context(featured)
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(featured)),
        "start_ct": featured["interval_start"].iloc[0].tz_convert("America/Chicago").isoformat(),
        "end_ct": featured["interval_start"].iloc[-1].tz_convert("America/Chicago").isoformat(),
        "zone_weights": {key: float(value) for key, value in weights.items()},
        "selection": selection,
        "metrics_pooled": metric_tables["pooled"].to_dict(orient="records"),
        "metrics_by_fold": metric_tables["by_fold"].to_dict(orient="records"),
        "prices": context,
        "headline": _headline(metric_tables["pooled"]),
    }
    if save:
        FORECASTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIGURES.mkdir(parents=True, exist_ok=True)
        forecasts.to_parquet(FORECASTS_PATH, index=False)
        METRICS_PATH.write_text(json.dumps(_ready(metric_tables["pooled"].to_dict(orient="records")), indent=2))
        SUMMARY_PATH.write_text(json.dumps(_ready(summary), indent=2))
        save_all(featured, forecasts, metric_tables["pooled"], FIGURES)
        LOGGER.info("Wrote forecasts, metrics, and figures")
    summary["forecasts"] = forecasts
    summary["metrics"] = metric_tables["pooled"]
    summary["featured"] = featured
    return summary


def _headline(pooled: pd.DataFrame) -> dict:
    def pull(target: str, model: str) -> dict:
        row = pooled.loc[(pooled["target"] == target) & (pooled["model"] == model)]
        if row.empty:
            return {}
        record = row.iloc[0]
        return {
            "mae": float(record["mae"]),
            "rmse": float(record["rmse"]),
            "mape": float(record["mape"]),
            "skill_vs_naive_24": float(record["skill_vs_naive_24"]),
            "n": int(record["n"]),
        }

    return {
        "load_naive_24": pull(LOAD_TARGET, "seasonal_naive_24"),
        "load_naive_168": pull(LOAD_TARGET, "seasonal_naive_168"),
        "load_sarimax": pull(LOAD_TARGET, "sarimax"),
        "load_gbm_lagged": pull(LOAD_TARGET, "gbm_lagged_weather"),
        "load_gbm_realized": pull(LOAD_TARGET, "gbm_realized_weather"),
        "price_naive_24": pull(PRICE_TARGET, "seasonal_naive_24"),
        "price_naive_168": pull(PRICE_TARGET, "seasonal_naive_168"),
        "price_ols": pull(PRICE_TARGET, "ols_lagged"),
        "price_gbm_lagged": pull(PRICE_TARGET, "gbm_lagged_weather"),
        "price_gbm_realized": pull(PRICE_TARGET, "gbm_realized_load_weather"),
    }


def _weights_from_frame(frame: pd.DataFrame) -> pd.Series:
    from ercot_forecast.features import zone_load_weights

    return zone_load_weights(frame)


def _ready(value):
    if isinstance(value, dict):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _ready(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _ready(value.to_dict())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value
