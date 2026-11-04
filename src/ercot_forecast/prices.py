"""Price-spike summaries a power desk can read next to the load forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ercot_forecast.config import CENTRAL, EXPLANATION_FEATURES, PRICE_TARGET, URI_END, URI_START
from ercot_forecast.features import date_mask


def price_context(frame: pd.DataFrame) -> dict:
    """Event windows, deciles, and a robust regression of the Houston hub price."""
    uri = window_stats(frame, URI_START, URI_END)
    peak_load_week = rolling_window(frame, "load_ercot", 168)
    peak_price_week = rolling_window(frame, "rtm_hb_houston_mean", 168)
    outside_uri = ~date_mask(frame, URI_START, URI_END)
    summer = frame["month"].isin([6, 7, 8])
    winter = frame["month"].isin([12, 1, 2])
    return {
        "uri": uri,
        "peak_load_hour": hour_record(frame, "load_ercot"),
        "peak_dam_hour": hour_record(frame, "dam_hb_houston"),
        "peak_rtm_hour": hour_record(frame, "rtm_hb_houston_max"),
        "peak_load_week": peak_load_week,
        "peak_rtm_week": peak_price_week,
        "negative_dam_hours": int((frame["dam_hb_houston"] < 0).sum()),
        "negative_dam_share": float((frame["dam_hb_houston"] < 0).mean()),
        "rtm_hours_ge_100": int((frame["rtm_hb_houston_max"] >= 100).sum()),
        "rtm_hours_ge_1000": int((frame["rtm_hb_houston_max"] >= 1000).sum()),
        "rtm_hours_ge_9000": int((frame["rtm_hb_houston_max"] >= 9000).sum()),
        "spikes_by_year": spikes_by_year(frame),
        "correlations": {
            "load_temp_f_weighted": _corr(frame["load_ercot"], frame["temp_f_weighted"]),
            "load_cdd_weighted": _corr(frame["load_ercot"], frame["cdd_weighted"]),
            "load_hdd_weighted": _corr(frame["load_ercot"], frame["hdd_weighted"]),
            "dam_load": _corr(frame["dam_hb_houston"], frame["load_ercot"]),
            "rtm_load": _corr(frame["rtm_hb_houston_mean"], frame["load_ercot"]),
            "dam_wind_fwest": _corr(frame["dam_hb_houston"], frame["wind_mph_fwest"]),
        },
        "deciles": {
            "summer": load_deciles(frame.loc[summer]),
            "winter": load_deciles(frame.loc[winter]),
            "excluding_uri": load_deciles(frame.loc[outside_uri]),
        },
        "ols": {
            "all_hours": explain_price(frame),
            "dam_below_200": explain_price(frame.loc[frame["dam_hb_houston"] < 200]),
        },
    }


def window_stats(frame: pd.DataFrame, start: str, end: str) -> dict:
    sub = frame.loc[date_mask(frame, start, end)]
    if sub.empty:
        raise ValueError(f"No hours in {start} to {end}")
    peak_load_at = sub.loc[sub["load_ercot"].idxmax(), "interval_start"]
    peak_price_at = sub.loc[sub["rtm_hb_houston_max"].idxmax(), "interval_start"]
    return {
        "start": start,
        "end": end,
        "hours": int(len(sub)),
        "load_mean_mw": _float(sub["load_ercot"].mean()),
        "load_max_mw": _float(sub["load_ercot"].max()),
        "load_max_at_ct": _iso(peak_load_at),
        "temp_f_weighted_mean": _float(sub["temp_f_weighted"].mean()),
        "temp_f_weighted_min": _float(sub["temp_f_weighted"].min()),
        "temp_f_coast_min": _float(sub["temp_f_coast"].min()),
        "wind_mph_fwest_mean": _float(sub["wind_mph_fwest"].mean()),
        "dam_hb_houston_mean": _float(sub["dam_hb_houston"].mean()),
        "dam_hb_houston_max": _float(sub["dam_hb_houston"].max()),
        "dam_hb_hubavg_mean": _float(sub["dam_hb_hubavg"].mean()),
        "dam_hb_hubavg_max": _float(sub["dam_hb_hubavg"].max()),
        "rtm_hb_houston_mean": _float(sub["rtm_hb_houston_mean"].mean()),
        "rtm_hb_houston_max": _float(sub["rtm_hb_houston_max"].max()),
        "rtm_max_at_ct": _iso(peak_price_at),
        "rtm_hb_hubavg_max": _float(sub["rtm_hb_hubavg_max"].max()),
        "hours_rtm_ge_100": int((sub["rtm_hb_houston_max"] >= 100).sum()),
        "hours_rtm_ge_1000": int((sub["rtm_hb_houston_max"] >= 1000).sum()),
        "hours_rtm_ge_9000": int((sub["rtm_hb_houston_max"] >= 9000).sum()),
    }


def rolling_window(frame: pd.DataFrame, column: str, hours: int) -> dict:
    values = frame[column].to_numpy(dtype=float)
    if len(values) < hours:
        raise ValueError("Series is shorter than the requested window")
    window_mean = pd.Series(values).rolling(hours).mean().to_numpy()
    end = int(np.nanargmax(window_mean))
    start = end - hours + 1
    block = frame.iloc[start : end + 1]
    start_ts = block["interval_start"].iloc[0]
    end_ts = block["interval_start"].iloc[-1]
    return {
        "column": column,
        "hours": hours,
        "start_ct": _iso(start_ts),
        "end_ct": _iso(end_ts),
        "mean": _float(block[column].mean()),
        "load_mean_mw": _float(block["load_ercot"].mean()),
        "load_max_mw": _float(block["load_ercot"].max()),
        "temp_f_weighted_mean": _float(block["temp_f_weighted"].mean()),
        "temp_f_coast_max": _float(block["temp_f_coast"].max()),
        "wind_mph_fwest_mean": _float(block["wind_mph_fwest"].mean()),
        "dam_hb_houston_mean": _float(block["dam_hb_houston"].mean()),
        "dam_hb_houston_max": _float(block["dam_hb_houston"].max()),
        "rtm_hb_houston_mean": _float(block["rtm_hb_houston_mean"].mean()),
        "rtm_hb_houston_max": _float(block["rtm_hb_houston_max"].max()),
    }


def hour_record(frame: pd.DataFrame, column: str) -> dict:
    row = frame.loc[frame[column].idxmax()]
    return {
        "column": column,
        "at_ct": _iso(row["interval_start"]),
        "value": _float(row[column]),
        "load_mw": _float(row["load_ercot"]),
        "dam_hb_houston": _float(row["dam_hb_houston"]),
        "dam_hb_hubavg": _float(row["dam_hb_hubavg"]),
        "rtm_hb_houston_mean": _float(row["rtm_hb_houston_mean"]),
        "rtm_hb_houston_max": _float(row["rtm_hb_houston_max"]),
        "temp_f_weighted": _float(row["temp_f_weighted"]),
        "temp_f_coast": _float(row["temp_f_coast"]),
        "wind_mph_fwest": _float(row["wind_mph_fwest"]),
    }


def spikes_by_year(frame: pd.DataFrame) -> list[dict]:
    local_year = frame["interval_start"].dt.tz_convert(CENTRAL).dt.year
    rows = []
    for year, sub in frame.groupby(local_year):
        rows.append(
            {
                "year": int(year),
                "hours": int(len(sub)),
                "rtm_ge_100": int((sub["rtm_hb_houston_max"] >= 100).sum()),
                "rtm_ge_1000": int((sub["rtm_hb_houston_max"] >= 1000).sum()),
                "dam_ge_100": int((sub["dam_hb_houston"] >= 100).sum()),
                "dam_mean": _float(sub["dam_hb_houston"].mean()),
                "dam_median": _float(sub["dam_hb_houston"].median()),
                "rtm_max": _float(sub["rtm_hb_houston_max"].max()),
                "load_max_mw": _float(sub["load_ercot"].max()),
            }
        )
    return rows


def load_deciles(frame: pd.DataFrame) -> list[dict]:
    sub = frame.dropna(subset=["load_ercot", PRICE_TARGET]).copy()
    if len(sub) < 10:
        return []
    sub["decile"] = pd.qcut(sub["load_ercot"], 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, group in sub.groupby("decile"):
        rows.append(
            {
                "decile": int(decile),
                "hours": int(len(group)),
                "load_mw_mean": _float(group["load_ercot"].mean()),
                "load_mw_min": _float(group["load_ercot"].min()),
                "load_mw_max": _float(group["load_ercot"].max()),
                "dam_mean": _float(group["dam_hb_houston"].mean()),
                "dam_median": _float(group["dam_hb_houston"].median()),
                "dam_p95": _float(np.nanpercentile(group["dam_hb_houston"], 95)),
                "rtm_mean": _float(group["rtm_hb_houston_mean"].mean()),
                "rtm_max": _float(group["rtm_hb_houston_max"].max()),
            }
        )
    return rows


def explain_price(frame: pd.DataFrame) -> dict:
    """OLS of the Houston day-ahead price on load, degree-days, and wind.

    Coefficients are associations with HC1 standard errors. The fit is not
    a forecast: same-hour load is included on purpose.
    """
    columns = EXPLANATION_FEATURES + [PRICE_TARGET]
    sub = frame.dropna(subset=columns)
    y = sub[PRICE_TARGET].astype(float)
    x = sm.add_constant(sub[EXPLANATION_FEATURES].astype(float), has_constant="add")
    fit = sm.OLS(y, x).fit(cov_type="HC1")
    coefs = {}
    for name in fit.params.index:
        coefs[str(name)] = {
            "coef": _float(fit.params[name]),
            "se": _float(fit.bse[name]),
            "pvalue": _float(fit.pvalues[name]),
        }
    return {"n": int(fit.nobs), "r2": _float(fit.rsquared), "coefs": coefs}


def _corr(left: pd.Series, right: pd.Series) -> float:
    return _float(left.corr(right))


def _float(value) -> float:
    return float(value)


def _iso(timestamp: pd.Timestamp) -> str:
    stamped = pd.Timestamp(timestamp)
    if stamped.tzinfo is None:
        stamped = stamped.tz_localize("UTC")
    return stamped.tz_convert(CENTRAL).isoformat()
