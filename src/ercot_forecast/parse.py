"""Turn ERCOT hour-ending labels into timezone-aware interval starts.

ERCOT reports the hour that just finished. Hour ending 01:00 on a delivery
date is the clock hour that started at 00:00 Central. Hour ending 24:00 is
the last hour of the delivery date. On the November fallback, the repeated
01:00-02:00 hour is marked "DST" in the load archive and "Y" in the price
files. That second hour is standard time.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd

from ercot_forecast.config import CENTRAL

_LOAD_RE = re.compile(
    r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})"
    r"\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})(?P<dst>\s*DST)?",
    re.IGNORECASE,
)


def interval_start_from_hour_ending(
    delivery_date: pd.Timestamp,
    hour_ending: int,
    repeated: bool = False,
) -> pd.Timestamp:
    """Return the UTC timestamp when the ERCOT hour started."""
    hour = int(hour_ending)
    if hour < 1 or hour > 24:
        raise ValueError(f"Hour ending must be between 1 and 24, got {hour_ending}")
    day = pd.Timestamp(delivery_date)
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    day = day.normalize()
    end = day + timedelta(days=1) if hour == 24 else day + timedelta(hours=hour)
    start = end - timedelta(hours=1)
    local = _localize(start, repeated=repeated)
    return local.tz_convert("UTC")


def parse_load_hour_ending(value: object) -> pd.Timestamp:
    """Parse a native-load label such as '11/07/2021 02:00 DST'."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return _from_clock_timestamp(pd.Timestamp(value))
    text = str(value).strip()
    match = _LOAD_RE.search(text)
    if match is None:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Could not parse load hour ending: {value!r}")
        return _from_clock_timestamp(pd.Timestamp(parsed))
    delivery = pd.Timestamp(
        year=int(match.group("year")),
        month=int(match.group("month")),
        day=int(match.group("day")),
    )
    repeated = bool(match.group("dst"))
    return interval_start_from_hour_ending(delivery, int(match.group("hour")), repeated)


def _from_clock_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Interpret a clock timestamp as an hour-ending label, without a DST flag."""
    naive = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    if naive.hour == 0 and naive.minute == 0:
        return interval_start_from_hour_ending(naive.normalize() - timedelta(days=1), 24, False)
    return interval_start_from_hour_ending(naive.normalize(), int(naive.hour), False)


def parse_clock_hour(value: object) -> int:
    """Accept 1, 1.0, '01:00', or '24:00'."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        hour = int(value)
    else:
        text = str(value).strip()
        hour = int(text.split(":")[0]) if ":" in text else int(float(text))
    if hour < 1 or hour > 24:
        raise ValueError(f"Clock hour must be between 1 and 24, got {value!r}")
    return hour


def is_repeated_hour(flag: object) -> bool:
    return str(flag).strip().upper() in {"Y", "YES", "DST", "TRUE", "1"}


def _localize(naive: pd.Timestamp, repeated: bool) -> pd.Timestamp:
    # True selects daylight time when the local clock is ambiguous.
    # The repeated ERCOT hour is the standard-time occurrence.
    ambiguous = not repeated
    try:
        return naive.tz_localize(CENTRAL, ambiguous=ambiguous, nonexistent="raise")
    except Exception as exc:
        name = type(exc).__name__
        if "NonExistent" in name or "nonexistent" in str(exc).lower():
            return naive.tz_localize(CENTRAL, ambiguous=ambiguous, nonexistent="shift_forward")
        raise


def attach_interval_start(
    frame: pd.DataFrame,
    date_col: str,
    hour_col: str,
    repeated_col: str | None = None,
) -> pd.DataFrame:
    """Add an `interval_start` UTC column. Repeated labels are parsed once."""
    out = frame.copy()
    dates = pd.to_datetime(out[date_col])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()
    hours = out[hour_col].map(parse_clock_hour)
    if repeated_col is None:
        repeated = pd.Series(False, index=out.index)
    else:
        repeated = out[repeated_col].map(is_repeated_hour)
    keys = pd.DataFrame({"date": dates, "hour": hours.astype(int), "repeated": repeated.astype(bool)})
    unique = keys.drop_duplicates()
    parsed: dict[tuple[pd.Timestamp, int, bool], pd.Timestamp] = {}
    for row in unique.itertuples(index=False):
        parsed[(row.date, int(row.hour), bool(row.repeated))] = interval_start_from_hour_ending(
            row.date, int(row.hour), bool(row.repeated)
        )
    out["interval_start"] = [
        parsed[(date, int(hour), bool(flag))]
        for date, hour, flag in zip(keys["date"], keys["hour"], keys["repeated"])
    ]
    out["interval_start"] = pd.DatetimeIndex(out["interval_start"])
    return out
