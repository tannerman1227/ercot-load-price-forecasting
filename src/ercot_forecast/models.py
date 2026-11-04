"""Day-ahead baselines, SARIMAX, OLS, and gradient boosting."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ercot_forecast.config import (
    GBM_GRID,
    LOAD_GBM_LAGGED,
    LOAD_GBM_REALIZED,
    LOAD_TARGET,
    PRICE_GBM_LAGGED,
    PRICE_GBM_REALIZED,
    PRICE_OLS_LAGGED,
    PRICE_TARGET,
    SARIMAX_CANDIDATES,
    SARIMAX_EXOG,
    SARIMAX_TRAIN_HOURS,
    VALIDATION,
)
from ercot_forecast.features import date_mask
from ercot_forecast.metrics import regression_metrics

LOGGER = logging.getLogger(__name__)
CALENDAR_CATEGORICAL = ("hour", "dow", "month", "is_weekend", "is_holiday")


def cross_validate(frame: pd.DataFrame, folds: tuple[dict, ...]) -> tuple[pd.DataFrame, dict]:
    """Expanding-window day-ahead evaluation for load and the Houston hub price.

    Hyperparameters are chosen on `VALIDATION`, which ends before every test
    fold. Later folds are allowed to train on earlier test periods.
    """
    _assert_folds_follow_validation(frame, folds)
    val_mask = date_mask(frame, VALIDATION["test_start"], VALIDATION["test_end"])
    notes: list[str] = []

    load_gbm_lagged = _tune_gbm(frame, LOAD_GBM_LAGGED, LOAD_TARGET, val_mask)
    load_gbm_realized = _tune_gbm(frame, LOAD_GBM_REALIZED, LOAD_TARGET, val_mask)
    price_gbm_lagged = _tune_gbm(frame, PRICE_GBM_LAGGED, PRICE_TARGET, val_mask)
    price_gbm_realized = _tune_gbm(frame, PRICE_GBM_REALIZED, PRICE_TARGET, val_mask)
    sarimax_spec = _tune_sarimax(frame, LOAD_TARGET, SARIMAX_EXOG, val_mask, notes)

    pieces: list[pd.DataFrame] = []
    for fold in folds:
        test_mask = date_mask(frame, fold["test_start"], fold["test_end"])
        origin = frame.loc[test_mask, "interval_start"].min()
        train_mask = frame["interval_start"] < origin
        pieces.extend(
            _score_fold(
                frame,
                fold["name"],
                train_mask,
                test_mask,
                load_gbm_lagged["params"],
                load_gbm_realized["params"],
                price_gbm_lagged["params"],
                price_gbm_realized["params"],
                sarimax_spec,
                notes,
            )
        )
    forecasts = pd.concat(pieces, ignore_index=True)
    selection = {
        "validation_window": VALIDATION,
        "load_gbm_lagged_weather": load_gbm_lagged,
        "load_gbm_realized_weather": load_gbm_realized,
        "price_gbm_lagged_weather": price_gbm_lagged,
        "price_gbm_realized_load_weather": price_gbm_realized,
        "sarimax": sarimax_spec,
        "notes": notes,
    }
    return forecasts, selection


def seasonal_naive(series: pd.Series, lag: int) -> pd.Series:
    return series.shift(lag)


def fit_predict_gbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict,
) -> pd.Series:
    model_frame = train.dropna(subset=features + [target])
    model = _gbm(features, params)
    model.fit(_design(model_frame, features), model_frame[target].to_numpy(dtype=float))
    design = _design(test, features)
    ok = np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
    predicted = np.full(len(test), np.nan)
    if ok.any():
        predicted[np.asarray(ok)] = model.predict(design.loc[ok])
    return pd.Series(predicted, index=test.index)


def fit_predict_ols(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.Series:
    model_frame = train.dropna(subset=features + [target])
    x_train = sm.add_constant(model_frame[features].astype(float), has_constant="add")
    x_test = sm.add_constant(test[features].astype(float), has_constant="add")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
    fit = sm.OLS(model_frame[target].astype(float), x_train).fit()
    predicted = np.asarray(fit.predict(x_test), dtype=float)
    return pd.Series(predicted, index=test.index)


def sarimax_forecast_window(
    frame: pd.DataFrame,
    target: str,
    exog_cols: list[str],
    test_mask: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    train_hours: int = SARIMAX_TRAIN_HOURS,
) -> pd.Series:
    """Fit on the hours just before the block, then forecast each local day.

    After each day the realized target is appended with `refit=False`, so the
    next day's forecast is a 24-hour-ahead forecast rather than a one-step
    update. Exogenous values must already be known at the day-ahead origin
    (lagged weather and the calendar).
    """
    positions = np.flatnonzero(np.asarray(test_mask))
    if positions.size == 0:
        raise ValueError("SARIMAX test window is empty")
    start = int(positions[0])
    end = int(positions[-1]) + 1
    if not np.array_equal(positions, np.arange(start, end)):
        raise ValueError("SARIMAX test window is not a contiguous hourly block")
    train_start = start - train_hours
    if train_start < 0:
        raise ValueError("Not enough history to fit SARIMAX")

    y = frame[target].to_numpy(dtype=float)
    exog = frame[exog_cols].to_numpy(dtype=float)
    if not np.isfinite(y[train_start:end]).all():
        raise ValueError("SARIMAX target has missing hours in the fit window")
    if not np.isfinite(exog[train_start:end]).all():
        raise ValueError("SARIMAX exogenous data has missing hours in the fit window")

    model = SARIMAX(
        y[train_start:start],
        order=order,
        seasonal_order=seasonal_order,
        exog=exog[train_start:start],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=150, method="lbfgs")
    predicted = np.empty(end - start, dtype=float)
    cursor = start
    offset = 0
    dates = frame["date"].to_numpy()
    while cursor < end:
        day = dates[cursor]
        steps = 1
        while cursor + steps < end and dates[cursor + steps] == day:
            steps += 1
        forecast = result.get_forecast(steps=steps, exog=exog[cursor : cursor + steps])
        predicted[offset : offset + steps] = np.asarray(forecast.predicted_mean, dtype=float)
        result = result.append(y[cursor : cursor + steps], exog=exog[cursor : cursor + steps], refit=False)
        cursor += steps
        offset += steps

    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out.iloc[start:end] = predicted
    return out


def _score_fold(
    frame,
    fold_name,
    train_mask,
    test_mask,
    load_lagged_params,
    load_realized_params,
    price_lagged_params,
    price_realized_params,
    sarimax_spec,
    notes,
) -> list[pd.DataFrame]:
    train = frame.loc[train_mask]
    test = frame.loc[test_mask]
    pieces = []

    load_predictions = {
        "seasonal_naive_24": (seasonal_naive(frame[LOAD_TARGET], 24), "day_ahead"),
        "seasonal_naive_168": (seasonal_naive(frame[LOAD_TARGET], 168), "day_ahead"),
        "gbm_lagged_weather": (
            _aligned(frame, test, fit_predict_gbm(train, test, LOAD_GBM_LAGGED, LOAD_TARGET, load_lagged_params)),
            "day_ahead",
        ),
        "gbm_realized_weather": (
            _aligned(frame, test, fit_predict_gbm(train, test, LOAD_GBM_REALIZED, LOAD_TARGET, load_realized_params)),
            "realized_weather",
        ),
    }
    try:
        load_predictions["sarimax"] = (
            sarimax_forecast_window(
                frame,
                LOAD_TARGET,
                SARIMAX_EXOG,
                test_mask,
                tuple(sarimax_spec["order"]),
                tuple(sarimax_spec["seasonal_order"]),
            ),
            "day_ahead",
        )
    except Exception as exc:
        notes.append(f"SARIMAX failed on {fold_name}: {exc}. That fold falls back to the 24-hour naive forecast.")
        LOGGER.warning("SARIMAX failed on %s: %s", fold_name, exc)
        load_predictions["sarimax"] = (seasonal_naive(frame[LOAD_TARGET], 24), "day_ahead")

    for name, (prediction, info_set) in load_predictions.items():
        clipped = prediction.clip(lower=0)
        pieces.append(_pack(frame, test_mask, LOAD_TARGET, name, info_set, clipped, fold_name))

    price_predictions = {
        "seasonal_naive_24": (seasonal_naive(frame[PRICE_TARGET], 24), "day_ahead"),
        "seasonal_naive_168": (seasonal_naive(frame[PRICE_TARGET], 168), "day_ahead"),
        "ols_lagged": (
            _aligned(frame, test, fit_predict_ols(train, test, PRICE_OLS_LAGGED, PRICE_TARGET)),
            "day_ahead",
        ),
        "gbm_lagged_weather": (
            _aligned(frame, test, fit_predict_gbm(train, test, PRICE_GBM_LAGGED, PRICE_TARGET, price_lagged_params)),
            "day_ahead",
        ),
        "gbm_realized_load_weather": (
            _aligned(frame, test, fit_predict_gbm(train, test, PRICE_GBM_REALIZED, PRICE_TARGET, price_realized_params)),
            "realized_load_and_weather",
        ),
    }
    for name, (prediction, info_set) in price_predictions.items():
        pieces.append(_pack(frame, test_mask, PRICE_TARGET, name, info_set, prediction, fold_name))
    return pieces


def _tune_gbm(frame: pd.DataFrame, features: list[str], target: str, val_mask: pd.Series) -> dict:
    origin = frame.loc[val_mask, "interval_start"].min()
    train = frame.loc[frame["interval_start"] < origin]
    test = frame.loc[val_mask]
    best: dict | None = None
    for params in GBM_GRID:
        prediction = fit_predict_gbm(train, test, features, target, params)
        if target == LOAD_TARGET:
            prediction = prediction.clip(lower=0)
        scores = regression_metrics(test[target], prediction)
        candidate = {"params": params, "validation_mae": scores["mae"], "validation_mape": scores["mape"]}
        if best is None or candidate["validation_mae"] < best["validation_mae"]:
            best = candidate
    assert best is not None
    LOGGER.info("Chose %s params for %s with validation MAE %.2f", best["params"], target, best["validation_mae"])
    return best


def _tune_sarimax(frame, target, exog_cols, val_mask, notes) -> dict:
    best: dict | None = None
    for order, seasonal_order in SARIMAX_CANDIDATES:
        try:
            prediction = sarimax_forecast_window(frame, target, exog_cols, val_mask, order, seasonal_order)
        except Exception as exc:
            notes.append(f"SARIMAX candidate {order}x{seasonal_order} failed on the validation window: {exc}")
            LOGGER.warning("SARIMAX candidate %s x %s failed: %s", order, seasonal_order, exc)
            continue
        scores = regression_metrics(frame.loc[val_mask, target], prediction.clip(lower=0).loc[val_mask])
        candidate = {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "validation_mae": scores["mae"],
            "validation_mape": scores["mape"],
        }
        LOGGER.info(
            "SARIMAX %s x %s validation MAE %.2f",
            order,
            seasonal_order,
            scores["mae"],
        )
        if best is None or candidate["validation_mae"] < best["validation_mae"]:
            best = candidate
    if best is None:
        order, seasonal_order = SARIMAX_CANDIDATES[0]
        best = {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "validation_mae": None,
            "validation_mape": None,
        }
        notes.append("Every SARIMAX candidate failed on the validation window. The first specification is kept.")
    return best


def _gbm(features: list[str], params: dict) -> HistGradientBoostingRegressor:
    categorical = [features.index(column) for column in CALENDAR_CATEGORICAL if column in features]
    return HistGradientBoostingRegressor(
        random_state=42,
        early_stopping=False,
        categorical_features=categorical,
        **params,
    )


def _design(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    design = frame[features].copy()
    for column in CALENDAR_CATEGORICAL:
        if column in design.columns:
            design[column] = design[column].astype("int64")
    return design


def _aligned(frame: pd.DataFrame, test: pd.DataFrame, prediction: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out.loc[test.index] = prediction.to_numpy(dtype=float)
    return out


def _pack(frame, test_mask, target, model, info_set, prediction, fold_name) -> pd.DataFrame:
    block = pd.DataFrame(
        {
            "interval_start": frame.loc[test_mask, "interval_start"].to_numpy(),
            "fold": fold_name,
            "target": target,
            "model": model,
            "info_set": info_set,
            "y_true": frame.loc[test_mask, target].to_numpy(dtype=float),
            "y_pred": prediction.loc[test_mask].to_numpy(dtype=float),
        }
    )
    return block.dropna(subset=["y_true", "y_pred"])


def _assert_folds_follow_validation(frame: pd.DataFrame, folds: tuple[dict, ...]) -> None:
    del frame
    for fold in folds:
        if fold["test_start"] < VALIDATION["test_end"]:
            raise ValueError(f"Fold {fold['name']} starts before the validation window ends")
