"""Download public ERCOT load, hub prices, and Open-Meteo weather.

Nothing here needs an API key. If a live request fails and raw files are
already on disk, the build continues from those files. The processed
parquet committed with the repo is the offline cache.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

from ercot_forecast.config import (
    CENTRAL,
    DAM_REPORT_TYPE_ID,
    DATA_RAW,
    DATASET_META_PATH,
    ERCOT_DOC_DOWNLOAD,
    ERCOT_DOC_LIST,
    HOURLY_PATH,
    HUBS,
    LOAD_HISTORY_PAGE,
    LOAD_URLS,
    OPEN_METEO_ARCHIVE,
    RTM_REPORT_TYPE_ID,
    YEARS,
    ZONES,
)
from ercot_forecast.features import zone_load_weights
from ercot_forecast.parse import attach_interval_start, parse_load_hour_ending

LOGGER = logging.getLogger(__name__)
HEADERS = {
    "User-Agent": "ercot-load-price-forecasting/0.1 (educational portfolio; public data)",
}
LOAD_COLUMNS = {
    "COAST": "load_coast",
    "EAST": "load_east",
    "FWEST": "load_fwest",
    "FAR WEST": "load_fwest",
    "FAR_WEST": "load_fwest",
    "NORTH": "load_north",
    "NCENT": "load_ncent",
    "NORTH CENTRAL": "load_ncent",
    "SOUTH": "load_south",
    "SOUTHERN": "load_south",
    "SCENT": "load_scent",
    "SOUTH CENTRAL": "load_scent",
    "WEST": "load_west",
    "ERCOT": "load_ercot",
}


def build_dataset(years: tuple[int, ...] = YEARS, refresh: bool = False) -> pd.DataFrame:
    """Download what is missing, join the hourly frame, and write the cache."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    HOURLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    load = _load_all_years(years, refresh=refresh)
    prices = _prices_all_years(years, refresh=refresh)
    weather = _weather_all_years(years, refresh=refresh)
    frame = load.merge(prices, on="interval_start", how="left").merge(weather, on="interval_start", how="left")
    frame = frame.sort_values("interval_start").drop_duplicates("interval_start")
    full_index = pd.date_range(frame["interval_start"].iloc[0], frame["interval_start"].iloc[-1], freq="h", tz="UTC")
    inserted = int(len(full_index) - len(frame))
    frame = frame.set_index("interval_start").reindex(full_index)
    frame.index.name = "interval_start"
    frame = frame.reset_index()
    weights = zone_load_weights(frame)
    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": list(years),
        "rows": int(len(frame)),
        "inserted_missing_hours": inserted,
        "start_utc": frame["interval_start"].iloc[0].isoformat(),
        "end_utc": frame["interval_start"].iloc[-1].isoformat(),
        "start_ct": frame["interval_start"].iloc[0].tz_convert(CENTRAL).isoformat(),
        "end_ct": frame["interval_start"].iloc[-1].tz_convert(CENTRAL).isoformat(),
        "zone_weights": {zone: float(weights[zone]) for zone in weights.index},
        "zone_cities": {zone: ZONES[zone]["city"] for zone in ZONES},
        "null_share": {column: float(frame[column].isna().mean()) for column in frame.columns if column != "interval_start"},
        "sources": {
            "load": "ERCOT hourly native load archives, https://www.ercot.com/gridinfo/load/load_hist",
            "dam": "ERCOT NP4-180-ER historical DAM load-zone and hub settlement point prices",
            "rtm": "ERCOT NP6-785-ER historical RTM load-zone and hub settlement point prices",
            "weather": "Open-Meteo historical weather API (archive-api.open-meteo.com), UTC hourly",
        },
    }
    frame.to_parquet(HOURLY_PATH, index=False)
    DATASET_META_PATH.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Wrote %s rows to %s", len(frame), HOURLY_PATH)
    return frame


def load_hourly_frame() -> pd.DataFrame:
    if not HOURLY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HOURLY_PATH}. Run `python scripts/download_data.py` on a machine with network access."
        )
    frame = pd.read_parquet(HOURLY_PATH)
    if frame["interval_start"].dt.tz is None:
        frame["interval_start"] = frame["interval_start"].dt.tz_localize("UTC")
    return frame.sort_values("interval_start").reset_index(drop=True)


def load_zone_weights() -> pd.Series:
    meta = json.loads(DATASET_META_PATH.read_text())
    return pd.Series(meta["zone_weights"], dtype=float)


def _load_all_years(years: tuple[int, ...], refresh: bool) -> pd.DataFrame:
    frames = [_read_load_year(year, refresh=refresh) for year in years]
    out = pd.concat(frames, ignore_index=True)
    out = out.groupby("interval_start", as_index=False).mean(numeric_only=True)
    return out.sort_values("interval_start")


def _read_load_year(year: int, refresh: bool) -> pd.DataFrame:
    path = _ensure_zip(f"native_load_{year}.zip", _load_url(year), refresh=refresh)
    workbook = _excel_from_zip(path)
    frame = _read_excel(workbook)
    renamed = {column: LOAD_COLUMNS[str(column).strip().upper()] for column in frame.columns if str(column).strip().upper() in LOAD_COLUMNS}
    if "load_ercot" not in renamed.values():
        raise ValueError(f"Load file for {year} is missing the ERCOT total column: {list(frame.columns)}")
    label_col = _load_label_column(frame)
    starts = []
    kept_positions = []
    for position, value in enumerate(frame[label_col].tolist()):
        if pd.isna(value):
            continue
        try:
            starts.append(parse_load_hour_ending(value))
        except (ValueError, TypeError):
            continue
        kept_positions.append(position)
    dropped = len(frame) - len(kept_positions)
    if dropped:
        LOGGER.warning("Dropped %s unparsed load rows in %s", dropped, year)
    parsed = frame.iloc[kept_positions].copy()
    parsed["interval_start"] = pd.DatetimeIndex(starts)
    for source, target in renamed.items():
        parsed[target] = pd.to_numeric(parsed[source], errors="coerce")
    parsed = parsed[["interval_start", *renamed.values()]]
    expected = 8784 if _is_leap(year) else 8760
    if len(parsed) != expected:
        LOGGER.warning("Load year %s has %s rows; expected %s", year, len(parsed), expected)
    return parsed


def _load_label_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).strip().lower() in {"hour ending", "hour_ending", "hourending"}:
            return column
    raise ValueError(f"No Hour Ending column in {list(frame.columns)}")


def _load_url(year: int) -> str:
    try:
        response = requests.get(LOAD_HISTORY_PAGE, headers=HEADERS, timeout=60)
        response.raise_for_status()
        found = {
            int(match.group(2)): match.group(1)
            for match in re.finditer(
                r'href="(https://www\.ercot\.com/files/docs/[^"]*Native_Load_(\d{4})[^"]*)"',
                response.text,
                flags=re.IGNORECASE,
            )
        }
        if year in found:
            return found[year]
    except requests.RequestException as exc:
        LOGGER.warning("Could not scrape the load archive page (%s). Using the stored URL.", exc)
    if year not in LOAD_URLS:
        raise KeyError(f"No load URL for {year}")
    return LOAD_URLS[year]


def _prices_all_years(years: tuple[int, ...], refresh: bool) -> pd.DataFrame:
    dam = pd.concat([_read_dam_year(year, refresh=refresh) for year in years], ignore_index=True)
    rtm = pd.concat([_read_rtm_year(year, refresh=refresh) for year in years], ignore_index=True)
    dam = _collapse_prices(dam)
    rtm = _collapse_prices(rtm)
    return dam.merge(rtm, on="interval_start", how="outer")


def _collapse_prices(frame: pd.DataFrame) -> pd.DataFrame:
    max_columns = [column for column in frame.columns if column.endswith("_max")]
    aggregations = {
        column: "max" if column in max_columns else "mean"
        for column in frame.columns
        if column != "interval_start"
    }
    return frame.groupby("interval_start", as_index=False).agg(aggregations)


def _read_dam_year(year: int, refresh: bool) -> pd.DataFrame:
    path = _ensure_market_zip("dam", year, DAM_REPORT_TYPE_ID, refresh=refresh)
    workbook = _excel_from_zip(path)
    sheets = _read_excel_sheets(workbook)
    pieces = []
    for sheet in sheets.values():
        columns = {str(column).strip().lower(): column for column in sheet.columns}
        if "settlement point" not in columns or "settlement point price" not in columns:
            continue
        point_col = columns["settlement point"]
        kept = sheet[sheet[point_col].isin(HUBS)].copy()
        if kept.empty:
            continue
        stamped = attach_interval_start(
            kept,
            date_col=columns["delivery date"],
            hour_col=columns["hour ending"],
            repeated_col=columns.get("repeated hour flag"),
        )
        stamped = stamped.rename(columns={point_col: "settlement_point", columns["settlement point price"]: "price"})
        pieces.append(stamped[["interval_start", "settlement_point", "price"]])
    long = pd.concat(pieces, ignore_index=True)
    long["price"] = pd.to_numeric(long["price"], errors="coerce")
    pivot = long.pivot_table(index="interval_start", columns="settlement_point", values="price", aggfunc="mean")
    return _rename_price_columns(pivot, prefix="dam", how="mean").reset_index()


def _read_rtm_year(year: int, refresh: bool) -> pd.DataFrame:
    path = _ensure_market_zip("rtm", year, RTM_REPORT_TYPE_ID, refresh=refresh)
    workbook = _excel_from_zip(path)
    sheets = _read_excel_sheets(workbook)
    pieces = []
    for sheet in sheets.values():
        columns = {str(column).strip().lower(): column for column in sheet.columns}
        point_key = "settlement point name" if "settlement point name" in columns else "settlement point"
        if point_key not in columns or "settlement point price" not in columns:
            continue
        point_col = columns[point_key]
        kept = sheet[sheet[point_col].isin(HUBS)].copy()
        if kept.empty:
            continue
        hour_col = columns.get("delivery hour", columns.get("hour ending"))
        stamped = attach_interval_start(
            kept,
            date_col=columns["delivery date"],
            hour_col=hour_col,
            repeated_col=columns.get("repeated hour flag"),
        )
        stamped = stamped.rename(columns={point_col: "settlement_point", columns["settlement point price"]: "price"})
        pieces.append(stamped[["interval_start", "settlement_point", "price"]])
    long = pd.concat(pieces, ignore_index=True)
    long["price"] = pd.to_numeric(long["price"], errors="coerce")
    grouped = long.groupby(["interval_start", "settlement_point"])["price"].agg(["mean", "max"])
    grouped = grouped.reset_index()
    mean_pivot = grouped.pivot(index="interval_start", columns="settlement_point", values="mean")
    max_pivot = grouped.pivot(index="interval_start", columns="settlement_point", values="max")
    mean_frame = _rename_price_columns(mean_pivot, prefix="rtm", how="mean")
    max_frame = _rename_price_columns(max_pivot, prefix="rtm", how="max")
    # The max series is the spike view. Keep Houston and the hub average.
    keep_max = [column for column in max_frame.columns if column.endswith("_max") and ("hb_houston" in column or "hb_hubavg" in column)]
    return mean_frame.merge(max_frame[keep_max], left_index=True, right_index=True, how="outer").reset_index()


def _rename_price_columns(pivot: pd.DataFrame, prefix: str, how: str) -> pd.DataFrame:
    rename = {}
    for column in pivot.columns:
        slug = str(column).strip().lower()
        suffix = "" if how == "mean" and prefix == "dam" else f"_{how}"
        if prefix == "dam":
            suffix = ""
        rename[column] = f"{prefix}_{slug}{suffix}"
    out = pivot.rename(columns=rename)
    return out


def _ensure_market_zip(market: str, year: int, report_type_id: int, refresh: bool) -> str:
    destination = DATA_RAW / f"{market}_{year}.zip"
    if destination.exists() and not refresh:
        return str(destination)
    docs = _document_list(report_type_id)
    match = next((doc for doc in docs if str(doc.get("FriendlyName", "")).endswith(str(year))), None)
    if match is None:
        if destination.exists():
            LOGGER.warning("No live %s file for %s. Using %s", market, year, destination)
            return str(destination)
        raise FileNotFoundError(f"No {market} archive for {year} in ERCOT report type {report_type_id}")
    url = ERCOT_DOC_DOWNLOAD.format(doc_id=match["DocID"])
    _download(url, destination)
    return str(destination)


def _document_list(report_type_id: int) -> list[dict]:
    url = ERCOT_DOC_LIST.format(report_type_id=report_type_id)
    payload = json.loads(_get_bytes(url))
    documents = payload["ListDocsByRptTypeRes"]["DocumentList"]
    return [item["Document"] for item in documents]


def _weather_all_years(years: tuple[int, ...], refresh: bool) -> pd.DataFrame:
    start = f"{min(years)}-01-01"
    # Hour ending 24:00 on Dec 31 is 23:00 Central, which falls on Jan 1 UTC.
    end = f"{max(years) + 1}-01-01"
    pieces = []
    for zone, point in ZONES.items():
        cache = DATA_RAW / f"weather_{zone}_{start}_{end}.csv"
        if cache.exists() and not refresh:
            frame = pd.read_csv(cache, parse_dates=["interval_start"])
        else:
            frame = _fetch_weather(float(point["lat"]), float(point["lon"]), start, end)
            frame.to_csv(cache, index=False)
        frame["interval_start"] = pd.to_datetime(frame["interval_start"], utc=True)
        frame = frame.rename(
            columns={
                "temperature_2m": f"temp_c_{zone}",
                "wind_speed_10m": f"wind_kph_{zone}",
            }
        )
        pieces.append(frame)
    weather = pieces[0]
    for piece in pieces[1:]:
        weather = weather.merge(piece, on="interval_start", how="outer")
    weather["interval_start"] = weather["interval_start"].dt.floor("h")
    weather = weather.groupby("interval_start", as_index=False).mean(numeric_only=True)
    return weather


def _fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    # The archive endpoint is reliable in one-year slices. Concatenate them.
    frames = []
    final_year = int(end[:4])
    final_month_day = end[5:]
    for year in range(int(start[:4]), final_year + 1):
        year_end = f"{year}-{final_month_day}" if year == final_year else f"{year}-12-31"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": year_end,
            "hourly": "temperature_2m,wind_speed_10m",
            "timezone": "UTC",
        }
        response = requests.get(OPEN_METEO_ARCHIVE, params=params, headers=HEADERS, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if "hourly" not in payload:
            raise RuntimeError(f"Open-Meteo returned no hourly data for {year}: {payload}")
        hourly = pd.DataFrame(payload["hourly"])
        frames.append(hourly)
        time.sleep(0.2)
    out = pd.concat(frames, ignore_index=True)
    out["interval_start"] = pd.to_datetime(out["time"], utc=True)
    return out[["interval_start", "temperature_2m", "wind_speed_10m"]]


def _ensure_zip(filename: str, url: str, refresh: bool) -> str:
    destination = DATA_RAW / filename
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return str(destination)
    _download(url, destination)
    return str(destination)


def _download(url: str, destination) -> None:
    LOGGER.info("Downloading %s", url)
    destination.write_bytes(_get_bytes(url))


def _get_bytes(url: str, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=HEADERS, timeout=180)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def _excel_from_zip(path: str) -> io.BytesIO:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".xlsx", ".xlsm", ".xls"))]
        if not names:
            raise FileNotFoundError(f"No spreadsheet inside {path}")
        return io.BytesIO(archive.read(names[0]))


def _read_excel(workbook: io.BytesIO) -> pd.DataFrame:
    workbook.seek(0)
    return pd.read_excel(workbook, engine="calamine")


def _read_excel_sheets(workbook: io.BytesIO) -> dict[str, pd.DataFrame]:
    workbook.seek(0)
    return pd.read_excel(workbook, sheet_name=None, engine="calamine")


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
