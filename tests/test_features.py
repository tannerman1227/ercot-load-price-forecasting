import numpy as np
import pandas as pd

from ercot_forecast.config import LOAD_GBM_LAGGED, LOAD_GBM_REALIZED, ZONES
from ercot_forecast.features import (
    build_features,
    cooling_degrees,
    date_mask,
    heating_degrees,
    zone_load_weights,
)


def test_degree_days_use_a_65f_base():
    assert cooling_degrees(80) == 15
    assert heating_degrees(80) == 0
    assert cooling_degrees(65) == 0
    assert heating_degrees(50) == 15
    assert cooling_degrees([60, 70]).tolist() == [0, 5]


def test_lags_are_prior_hours_not_the_current_hour():
    frame = _hourly_frame(hours=200)
    weights = zone_load_weights(frame)
    featured = build_features(frame, weights)
    assert featured.loc[24, "load_ercot_lag_24"] == featured.loc[0, "load_ercot"]
    assert featured.loc[168, "load_ercot_lag_168"] == featured.loc[0, "load_ercot"]
    assert np.isnan(featured.loc[10, "load_ercot_lag_24"])


def test_operational_load_features_do_not_see_the_target_hour():
    for name in LOAD_GBM_LAGGED:
        assert name != "load_ercot"
        if name.startswith(("temp_", "cdd_", "hdd_", "wind_", "load_")):
            assert "lag_" in name
    assert "temp_f_weighted" in LOAD_GBM_REALIZED
    assert "load_ercot" not in LOAD_GBM_REALIZED


def test_holiday_and_weekend_flags():
    frame = _hourly_frame(start="2024-07-04", hours=72)
    featured = build_features(frame, zone_load_weights(frame))
    july4 = featured.loc[featured["date"].astype(str) == "2024-07-04", "is_holiday"]
    july5 = featured.loc[featured["date"].astype(str) == "2024-07-05", "is_holiday"]
    assert july4.eq(1).all()
    assert july5.eq(0).all()
    assert featured.loc[featured["dow"] >= 5, "is_weekend"].eq(1).all()


def test_date_mask_is_half_open_in_central_time():
    frame = _hourly_frame(start="2024-07-15", hours=72)
    featured = build_features(frame, zone_load_weights(frame))
    mask = date_mask(featured, "2024-07-15", "2024-07-16")
    assert int(mask.sum()) == 24


def test_zone_weights_sum_to_one():
    weights = zone_load_weights(_hourly_frame(hours=48))
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert set(weights.index) == set(ZONES)


def _hourly_frame(hours: int = 240, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    frame = pd.DataFrame({"interval_start": index})
    frame["load_ercot"] = np.arange(hours, dtype=float) + 30000
    frame["dam_hb_houston"] = 20 + np.arange(hours) * 0.01
    for zone in ZONES:
        frame[f"load_{zone}"] = frame["load_ercot"] / len(ZONES)
        frame[f"temp_c_{zone}"] = 20 + (index.hour.to_numpy() - 12)
        frame[f"wind_kph_{zone}"] = 15.0
    return frame


def test_rolling_lag_matches_the_prior_week():
    frame = _hourly_frame(hours=400)
    featured = build_features(frame, zone_load_weights(frame))
    row = 300
    expected = featured.loc[row - 24 - 167 : row - 24, "load_ercot"].mean()
    assert featured.loc[row, "load_ercot_roll168_lag_24"] == expected
