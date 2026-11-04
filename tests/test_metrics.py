import numpy as np
import pandas as pd
import pytest

from ercot_forecast.metrics import regression_metrics, skill_score, summarize_forecasts


def test_perfect_forecast_has_zero_error():
    actual = np.array([100.0, 200.0, 300.0])
    scores = regression_metrics(actual, actual)
    assert scores["mae"] == 0
    assert scores["rmse"] == 0
    assert scores["mape"] == 0
    assert scores["n"] == 3


def test_known_errors_and_mape_floor():
    scores = regression_metrics([100.0, 0.0, 50.0], [110.0, 5.0, 40.0], mape_floor=1.0)
    assert scores["mae"] == (10 + 5 + 10) / 3
    assert scores["n_mape"] == 2
    assert abs(scores["mape"] - ((10 / 100 + 10 / 50) / 2) * 100) < 1e-9
    assert abs(scores["rmse"] - np.sqrt((100 + 25 + 100) / 3)) < 1e-9


def test_skill_score_is_positive_when_the_model_is_closer():
    assert skill_score(80, 100) == pytest.approx(0.2)
    assert np.isnan(skill_score(10, 0))


def test_summary_reports_skill_against_the_24h_naive():
    forecasts = pd.DataFrame(
        {
            "fold": ["a", "a"],
            "target": ["load_ercot", "load_ercot"],
            "model": ["seasonal_naive_24", "gbm_lagged_weather"],
            "info_set": ["day_ahead", "day_ahead"],
            "y_true": [100.0, 100.0],
            "y_pred": [110.0, 104.0],
        }
    )
    pooled = summarize_forecasts(forecasts)["pooled"]
    gbm = pooled.loc[pooled["model"] == "gbm_lagged_weather"].iloc[0]
    assert gbm["mae"] == 4
    assert abs(gbm["skill_vs_naive_24"] - 0.6) < 1e-9
