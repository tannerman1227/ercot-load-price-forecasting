import numpy as np
import pandas as pd

from ercot_forecast.prices import explain_price, load_deciles, window_stats


def test_window_stats_and_deciles_use_the_supplied_hours():
    index = pd.date_range("2021-02-13", periods=48, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "interval_start": index,
            "month": index.tz_convert("America/Chicago").month,
            "load_ercot": 45000 + 8000 * np.sin(np.linspace(0, 9, 48)),
            "dam_hb_houston": 25 + 10 * np.sin(np.linspace(0, 4, 48)) + np.linspace(0, 30, 48),
            "dam_hb_hubavg": 24 + 8 * np.cos(np.linspace(0, 4, 48)),
            "rtm_hb_houston_mean": 30 + 20 * np.sin(np.linspace(0, 5, 48)),
            "rtm_hb_houston_max": 40 + np.linspace(0, 500, 48),
            "rtm_hb_hubavg_max": 40 + np.linspace(0, 400, 48),
            "temp_f_weighted": 40 + 25 * np.sin(np.linspace(0, 6, 48)),
            "temp_f_coast": 38 + 20 * np.sin(np.linspace(0, 6, 48)),
            "wind_mph_fwest": 8 + 6 * np.cos(np.linspace(0, 5, 48)),
            "cdd_weighted": np.maximum(np.sin(np.linspace(0, 6, 48)), 0) * 12,
            "hdd_weighted": np.maximum(-np.sin(np.linspace(0, 6, 48)), 0) * 15,
            "hour_sin": np.sin(np.linspace(0, 8, 48)),
            "hour_cos": np.cos(np.linspace(0, 8, 48)),
            "is_weekend": np.array([i % 7 >= 5 for i in range(48)], dtype=int),
            "load_gw": (45000 + 8000 * np.sin(np.linspace(0, 9, 48))) / 1000,
        }
    )
    stats = window_stats(frame, "2021-02-13", "2021-02-14")
    assert stats["hours"] == 24
    assert stats["load_max_mw"] > stats["load_mean_mw"]
    assert stats["rtm_hb_houston_max"] > 100
    deciles = load_deciles(frame)
    assert deciles[0]["decile"] == 1
    assert deciles[-1]["load_mw_mean"] > deciles[0]["load_mw_mean"]
    fit = explain_price(frame)
    assert fit["n"] == 48
    assert "load_gw" in fit["coefs"]
