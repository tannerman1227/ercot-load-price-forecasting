import json

import pandas as pd
import pytest

from ercot_forecast.config import CENTRAL, DATASET_META_PATH, HOURLY_PATH


def test_cached_hourly_frame_is_complete_and_contains_uri():
    if not HOURLY_PATH.exists():
        pytest.skip("Processed cache is built by scripts/download_data.py")
    frame = pd.read_parquet(HOURLY_PATH)
    if frame["interval_start"].dt.tz is None:
        frame["interval_start"] = frame["interval_start"].dt.tz_localize("UTC")
    assert frame["interval_start"].is_monotonic_increasing
    diffs = frame["interval_start"].diff().dt.total_seconds().dropna()
    assert (diffs == 3600).all()
    for column in ("load_ercot", "dam_hb_houston", "rtm_hb_houston_mean", "temp_c_coast", "wind_kph_fwest"):
        assert column in frame.columns
        assert frame[column].notna().mean() > 0.98
    assert frame["load_ercot"].min() > 20000
    local = frame["interval_start"].dt.tz_convert(CENTRAL)
    uri = frame[(local >= "2021-02-13") & (local < "2021-02-20")]
    assert uri["rtm_hb_houston_max"].max() > 1000
    assert uri["load_ercot"].max() > 50000
    meta = json.loads(DATASET_META_PATH.read_text())
    assert abs(sum(meta["zone_weights"].values()) - 1) < 1e-9
    assert meta["rows"] == len(frame)
