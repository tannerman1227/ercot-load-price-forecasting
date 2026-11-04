import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ercot_forecast.models import fit_predict_gbm, sarimax_forecast_window, seasonal_naive


def test_seasonal_naive_shifts_by_the_requested_lag():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert seasonal_naive(series, 2).tolist()[2:] == [1.0, 2.0]


def test_gbm_learns_a_lagged_level():
    hour = np.tile(np.arange(24), 20)
    load = 30000 + 2500 * np.sin(2 * np.pi * hour / 24)
    frame = pd.DataFrame(
        {
            "load_ercot": load,
            "load_ercot_lag_24": pd.Series(load).shift(24),
            "hour": hour,
            "dow": np.zeros(len(hour), dtype=int),
            "month": np.ones(len(hour), dtype=int),
            "is_weekend": np.zeros(len(hour), dtype=int),
            "is_holiday": np.zeros(len(hour), dtype=int),
        }
    )
    features = ["load_ercot_lag_24", "hour", "dow", "month", "is_weekend", "is_holiday"]
    prediction = fit_predict_gbm(
        frame.iloc[:360],
        frame.iloc[360:],
        features,
        "load_ercot",
        {"max_depth": 4, "max_iter": 80, "learning_rate": 0.1, "min_samples_leaf": 5, "l2_regularization": 0.0},
    )
    assert np.isfinite(prediction).all()
    mae = np.mean(np.abs(prediction.to_numpy() - frame.iloc[360:]["load_ercot"].to_numpy()))
    assert mae < 400


def test_sarimax_day_ahead_window_is_finite():
    index = pd.date_range("2024-06-01", periods=24 * 12, freq="h", tz="UTC")
    local = index.tz_convert("America/Chicago")
    load = 40000 + 3000 * np.sin(2 * np.pi * local.hour.to_numpy() / 24)
    frame = pd.DataFrame(
        {
            "interval_start": index,
            "load_ercot": load,
            "temp_f_weighted_lag_24": pd.Series(load).shift(24).fillna(70).to_numpy(),
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "date": local.date,
        }
    )
    test_mask = pd.Series(False, index=frame.index)
    test_mask.iloc[-48:] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        prediction = sarimax_forecast_window(
            frame,
            "load_ercot",
            ["temp_f_weighted_lag_24", "is_weekend"],
            test_mask,
            (1, 0, 0),
            (1, 0, 0, 24),
            train_hours=24 * 5,
        )
    assert np.isfinite(prediction.iloc[-48:]).all()
    assert prediction.iloc[:-48].isna().all()
