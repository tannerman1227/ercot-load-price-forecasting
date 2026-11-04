"""Calendar, degree-day, and lag features for a day-ahead horizon.

Lag features are shifted by a whole number of rows. The modeling frame is a
regular UTC hourly index, so a shift of 24 is the same clock hour yesterday,
including across the Central time daylight-saving transitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ercot_forecast.config import CENTRAL, DEGREE_DAY_BASE_F, ZONES


def celsius_to_fahrenheit(temp_c: pd.Series | np.ndarray | float) -> np.ndarray:
    return np.asarray(temp_c, dtype=float) * 9.0 / 5.0 + 32.0


def cooling_degrees(
    temp_f: pd.Series | np.ndarray | float,
    base_f: float = DEGREE_DAY_BASE_F,
) -> np.ndarray:
    """Degrees above the base. Hourly, not divided by 24."""
    return np.maximum(np.asarray(temp_f, dtype=float) - base_f, 0.0)


def heating_degrees(
    temp_f: pd.Series | np.ndarray | float,
    base_f: float = DEGREE_DAY_BASE_F,
) -> np.ndarray:
    """Degrees below the base. Hourly, not divided by 24."""
    return np.maximum(base_f - np.asarray(temp_f, dtype=float), 0.0)


def zone_load_weights(frame: pd.DataFrame) -> pd.Series:
    """Structural share of each weather zone in average ERCOT load."""
    columns = [f"load_{zone}" for zone in ZONES]
    means = frame[columns].mean()
    total = float(means.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Zone loads must have a positive mean to form weights")
    weights = means / total
    weights.index = list(ZONES)
    return weights


def build_features(frame: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Return a copy with weather, calendar, and lag columns added."""
    out = frame.sort_values("interval_start").reset_index(drop=True).copy()
    if out["interval_start"].dt.tz is None:
        out["interval_start"] = out["interval_start"].dt.tz_localize("UTC")
    else:
        out["interval_start"] = out["interval_start"].dt.tz_convert("UTC")

    local = out["interval_start"].dt.tz_convert(CENTRAL)
    out["hour"] = local.dt.hour.astype(int)
    out["dow"] = local.dt.dayofweek.astype(int)
    out["month"] = local.dt.month.astype(int)
    out["doy"] = local.dt.dayofyear.astype(int)
    out["date"] = local.dt.date
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_repeated_hour"] = [int(ts.fold == 1) for ts in local]
    out["is_holiday"] = _holiday_flags(local)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 366.0)

    for zone in ZONES:
        out[f"temp_f_{zone}"] = celsius_to_fahrenheit(out[f"temp_c_{zone}"])
        out[f"wind_mph_{zone}"] = out[f"wind_kph_{zone}"] / 1.609344

    out["temp_f_weighted"] = sum(out[f"temp_f_{zone}"] * float(weights[zone]) for zone in ZONES)
    out["wind_mph_weighted"] = sum(out[f"wind_mph_{zone}"] * float(weights[zone]) for zone in ZONES)
    out["cdd_weighted"] = cooling_degrees(out["temp_f_weighted"])
    out["hdd_weighted"] = heating_degrees(out["temp_f_weighted"])
    out["cdd_coast"] = cooling_degrees(out["temp_f_coast"])
    out["hdd_coast"] = heating_degrees(out["temp_f_coast"])
    out["load_gw"] = out["load_ercot"] / 1000.0

    out = _add_time_shifts(out)
    return out


def date_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    """True for Central-time hours in [start, end). Dates are YYYY-MM-DD."""
    local = frame["interval_start"].dt.tz_convert(CENTRAL)
    start_ts = pd.Timestamp(start, tz=CENTRAL)
    end_ts = pd.Timestamp(end, tz=CENTRAL)
    return (local >= start_ts) & (local < end_ts)


def _holiday_flags(local: pd.Series) -> pd.Series:
    import holidays

    years = sorted(set(local.dt.year.tolist()))
    us_holidays = holidays.US(years=years)
    return local.dt.date.map(lambda day: int(day in us_holidays)).astype(int)


def _add_time_shifts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    specs = {
        "load_ercot_lag_24": ("load_ercot", 24),
        "load_ercot_lag_48": ("load_ercot", 48),
        "load_ercot_lag_168": ("load_ercot", 168),
        "dam_hb_houston_lag_24": ("dam_hb_houston", 24),
        "dam_hb_houston_lag_168": ("dam_hb_houston", 168),
        "temp_f_weighted_lag_24": ("temp_f_weighted", 24),
        "temp_f_coast_lag_24": ("temp_f_coast", 24),
        "temp_f_ncent_lag_24": ("temp_f_ncent", 24),
        "temp_f_scent_lag_24": ("temp_f_scent", 24),
        "cdd_weighted_lag_24": ("cdd_weighted", 24),
        "hdd_weighted_lag_24": ("hdd_weighted", 24),
        "wind_mph_fwest_lag_24": ("wind_mph_fwest", 24),
        "wind_mph_coast_lag_24": ("wind_mph_coast", 24),
    }
    for name, (column, lag) in specs.items():
        out[name] = out[column].shift(lag)
    # Mean of the 168 hours ending 24 hours ago. Known at a day-ahead origin.
    out["load_ercot_roll168_lag_24"] = out["load_ercot"].rolling(168).mean().shift(24)
    out["dam_hb_houston_roll168_lag_24"] = out["dam_hb_houston"].rolling(168).mean().shift(24)
    return out
