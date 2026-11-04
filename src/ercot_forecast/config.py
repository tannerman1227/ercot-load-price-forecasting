"""Paths, weather-zone points, and the cross-validation calendar."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

HOURLY_PATH = DATA_PROCESSED / "ercot_hourly.parquet"
DATASET_META_PATH = DATA_PROCESSED / "dataset_meta.json"
FORECASTS_PATH = DATA_PROCESSED / "forecasts.parquet"
METRICS_PATH = DATA_PROCESSED / "metrics.json"
SUMMARY_PATH = DATA_PROCESSED / "summary.json"

CENTRAL = "America/Chicago"
DEGREE_DAY_BASE_F = 65.0

# Representative city in each ERCOT weather zone. These are not official
# zone centroids; Coast, North Central, and South Central are the large
# load centers (Houston, Dallas-Fort Worth, Austin).
ZONES: dict[str, dict[str, float | str]] = {
    "coast": {"city": "Houston", "lat": 29.7604, "lon": -95.3698},
    "east": {"city": "Tyler", "lat": 32.3513, "lon": -95.3011},
    "fwest": {"city": "Midland", "lat": 31.9973, "lon": -102.0779},
    "north": {"city": "Wichita Falls", "lat": 33.9137, "lon": -98.4934},
    "ncent": {"city": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "south": {"city": "Corpus Christi", "lat": 27.8006, "lon": -97.3964},
    "scent": {"city": "Austin", "lat": 30.2672, "lon": -97.7431},
    "west": {"city": "Abilene", "lat": 32.4487, "lon": -99.7331},
}

YEARS = (2021, 2022, 2023, 2024, 2025)

# Published ERCOT archives. The downloader prefers the links it finds on
# the load-history page and falls back to these.
LOAD_URLS = {
    2021: "https://www.ercot.com/files/docs/2021/11/12/Native_Load_2021.zip",
    2022: "https://www.ercot.com/files/docs/2022/02/08/Native_Load_2022.zip",
    2023: "https://www.ercot.com/files/docs/2023/02/09/Native_Load_2023.zip",
    2024: "https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip",
    2025: "https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip",
}
LOAD_HISTORY_PAGE = "https://www.ercot.com/gridinfo/load/load_hist"
DAM_REPORT_TYPE_ID = 13060  # NP4-180-ER historical DAM hub and load-zone SPPs
RTM_REPORT_TYPE_ID = 13061  # NP6-785-ER historical RTM hub and load-zone SPPs
ERCOT_DOC_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={report_type_id}"
ERCOT_DOC_DOWNLOAD = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc_id}"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

HUBS = ("HB_HOUSTON", "HB_HUBAVG", "HB_NORTH", "HB_WEST", "LZ_HOUSTON")

# Hyperparameters are chosen on this window only. Every test fold starts later.
VALIDATION = {"name": "validation_2024_06", "test_start": "2024-06-10", "test_end": "2024-06-24"}

FOLDS = (
    {"name": "summer_2024", "test_start": "2024-07-15", "test_end": "2024-07-29"},
    {"name": "winter_2024", "test_start": "2024-12-09", "test_end": "2024-12-23"},
    {"name": "spring_2025", "test_start": "2025-04-07", "test_end": "2025-04-21"},
    {"name": "summer_2025", "test_start": "2025-08-04", "test_end": "2025-08-18"},
    {"name": "fall_2025", "test_start": "2025-10-13", "test_end": "2025-10-27"},
)

# Trailing history used to fit each SARIMAX. Six weeks covers several
# weekly cycles without a full-sample hourly state-space fit.
SARIMAX_TRAIN_HOURS = 24 * 42
SARIMAX_CANDIDATES = (
    ((1, 0, 0), (1, 0, 0, 24)),
    ((1, 0, 1), (1, 0, 0, 24)),
    ((2, 0, 1), (1, 0, 0, 24)),
)

GBM_GRID = (
    {"max_depth": 4, "learning_rate": 0.08, "max_iter": 200, "min_samples_leaf": 40, "l2_regularization": 0.1},
    {"max_depth": 6, "learning_rate": 0.06, "max_iter": 300, "min_samples_leaf": 30, "l2_regularization": 0.1},
    {"max_depth": 8, "learning_rate": 0.05, "max_iter": 350, "min_samples_leaf": 20, "l2_regularization": 0.2},
)

# MAPE ignores hours with |actual| below this floor so a price printed at
# zero does not send the percentage to infinity. Load is far above it.
MAPE_FLOOR = 1.0

URI_START = "2021-02-13"
URI_END = "2021-02-20"  # exclusive, Central time

LOAD_TARGET = "load_ercot"
PRICE_TARGET = "dam_hb_houston"

LOAD_GBM_LAGGED = [
    "load_ercot_lag_24",
    "load_ercot_lag_48",
    "load_ercot_lag_168",
    "load_ercot_roll168_lag_24",
    "temp_f_weighted_lag_24",
    "temp_f_coast_lag_24",
    "temp_f_ncent_lag_24",
    "temp_f_scent_lag_24",
    "cdd_weighted_lag_24",
    "hdd_weighted_lag_24",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
]

LOAD_GBM_REALIZED = [
    "load_ercot_lag_24",
    "load_ercot_lag_48",
    "load_ercot_lag_168",
    "load_ercot_roll168_lag_24",
    "temp_f_weighted",
    "temp_f_coast",
    "temp_f_ncent",
    "temp_f_scent",
    "cdd_weighted",
    "hdd_weighted",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
]

SARIMAX_EXOG = ["temp_f_weighted_lag_24", "is_weekend"]

PRICE_GBM_LAGGED = [
    "dam_hb_houston_lag_24",
    "dam_hb_houston_lag_168",
    "dam_hb_houston_roll168_lag_24",
    "load_ercot_lag_24",
    "load_ercot_lag_168",
    "temp_f_weighted_lag_24",
    "cdd_weighted_lag_24",
    "hdd_weighted_lag_24",
    "wind_mph_fwest_lag_24",
    "wind_mph_coast_lag_24",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
]

PRICE_OLS_LAGGED = [
    "dam_hb_houston_lag_24",
    "load_ercot_lag_24",
    "cdd_weighted_lag_24",
    "hdd_weighted_lag_24",
    "wind_mph_fwest_lag_24",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_holiday",
]

PRICE_GBM_REALIZED = [
    "dam_hb_houston_lag_24",
    "dam_hb_houston_lag_168",
    "load_ercot",
    "temp_f_weighted",
    "cdd_weighted",
    "hdd_weighted",
    "wind_mph_fwest",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
]

EXPLANATION_FEATURES = [
    "load_gw",
    "cdd_weighted",
    "hdd_weighted",
    "wind_mph_fwest",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]

MODEL_LABELS = {
    "seasonal_naive_24": "Seasonal naive (24h)",
    "seasonal_naive_168": "Seasonal naive (168h)",
    "sarimax": "SARIMAX",
    "ols_lagged": "OLS",
    "gbm_lagged_weather": "Gradient boosting",
    "gbm_realized_weather": "GBM, realized weather",
    "gbm_realized_load_weather": "GBM, realized load and weather",
}
