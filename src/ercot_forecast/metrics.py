"""Point-forecast scores for load (MW) and price ($/MWh)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ercot_forecast.config import MAPE_FLOOR


def regression_metrics(
    y_true,
    y_pred,
    mape_floor: float = MAPE_FLOOR,
) -> dict[str, float]:
    """MAE, RMSE, and MAPE on the hours where both values are finite.

    MAPE uses |actual| as the denominator and drops hours below
    `mape_floor`, so a zero or near-zero price does not dominate it.
    MAE and RMSE still include those hours.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if actual.size == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "n_mape": 0,
            "mape_floor": float(mape_floor),
        }
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.abs(actual)
    mape_ok = denom >= mape_floor
    if mape_ok.any():
        mape = float(np.mean(np.abs(error[mape_ok]) / denom[mape_ok]) * 100.0)
    else:
        mape = np.nan
    return {
        "n": int(actual.size),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n_mape": int(mape_ok.sum()),
        "mape_floor": float(mape_floor),
    }


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """One minus the MAE ratio. Positive means the model beats the baseline."""
    if baseline_mae is None or not np.isfinite(baseline_mae) or baseline_mae == 0:
        return np.nan
    return float(1.0 - model_mae / baseline_mae)


def summarize_forecasts(forecasts: pd.DataFrame) -> dict:
    """Pooled and per-fold metrics for a long forecast table."""
    pooled = []
    by_fold = []
    group_cols = ["target", "model", "info_set"]
    for keys, group in forecasts.groupby(group_cols, sort=False):
        scores = regression_metrics(group["y_true"], group["y_pred"])
        row = dict(zip(group_cols, keys))
        row.update(scores)
        pooled.append(row)
        for fold, fold_group in group.groupby("fold", sort=False):
            fold_scores = regression_metrics(fold_group["y_true"], fold_group["y_pred"])
            fold_row = dict(zip(group_cols, keys))
            fold_row["fold"] = fold
            fold_row.update(fold_scores)
            by_fold.append(fold_row)
    pooled_df = pd.DataFrame(pooled)
    if not pooled_df.empty:
        baselines = pooled_df.loc[
            pooled_df["model"] == "seasonal_naive_24",
            ["target", "mae"],
        ].rename(columns={"mae": "baseline_mae"})
        pooled_df = pooled_df.merge(baselines, on="target", how="left")
        pooled_df["skill_vs_naive_24"] = [
            skill_score(mae, base) for mae, base in zip(pooled_df["mae"], pooled_df["baseline_mae"])
        ]
    return {"pooled": pooled_df, "by_fold": pd.DataFrame(by_fold)}
