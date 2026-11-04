import json

from ercot_forecast.config import FIGURES, FORECASTS_PATH, SUMMARY_PATH


def test_scored_study_is_checked_in():
    assert SUMMARY_PATH.exists()
    assert FORECASTS_PATH.exists()
    for name in (
        "load_vs_temperature.png",
        "load_forecast_overlay.png",
        "uri_february_2021.png",
        "peak_load_week.png",
        "price_vs_load.png",
    ):
        assert (FIGURES / name).exists()
        assert (FIGURES / name).stat().st_size > 10_000
    summary = json.loads(SUMMARY_PATH.read_text())
    load = summary["headline"]["load_gbm_lagged"]
    naive = summary["headline"]["load_naive_24"]
    assert load["n"] == 1680
    assert naive["n"] == 1680
    assert summary["prices"]["uri"]["rtm_hb_houston_max"] > 1000
    assert summary["prices"]["peak_load_hour"]["load_mw"] > summary["prices"]["uri"]["load_max_mw"]
