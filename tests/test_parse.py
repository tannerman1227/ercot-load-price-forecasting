import pandas as pd

from ercot_forecast.parse import interval_start_from_hour_ending, parse_load_hour_ending


def test_hour_ending_midnight_is_previous_evening():
    stamp = parse_load_hour_ending("01/01/2021 01:00")
    assert stamp == pd.Timestamp("2021-01-01 06:00", tz="UTC")
    stamp = parse_load_hour_ending("01/01/2021 24:00")
    assert stamp == pd.Timestamp("2021-01-02 05:00", tz="UTC")


def test_november_fallback_repeated_hour_is_standard_time():
    first = parse_load_hour_ending("11/07/2021 02:00")
    second = parse_load_hour_ending("11/07/2021 02:00 DST")
    assert first == pd.Timestamp("2021-11-07 06:00", tz="UTC")
    assert second == pd.Timestamp("2021-11-07 07:00", tz="UTC")
    assert second > first


def test_spring_forward_hours_stay_one_hour_apart():
    before = parse_load_hour_ending("03/14/2021 02:00")
    after = parse_load_hour_ending("03/14/2021 04:00")
    assert before == pd.Timestamp("2021-03-14 07:00", tz="UTC")
    assert after == pd.Timestamp("2021-03-14 08:00", tz="UTC")


def test_price_hour_ending_matches_load_label():
    price = interval_start_from_hour_ending(pd.Timestamp("2021-02-17"), 5, repeated=False)
    assert price == pd.Timestamp("2021-02-17 10:00", tz="UTC")
