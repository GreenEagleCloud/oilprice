"""Shared configuration for the Rhine (Basel-Rheinhalle + Kaub) collectors.

Used by:
    * src/collect_rhein_levels.py         (gauge / discharge collector)
    * src/collect_rhein_cargo.py          (Port-of-Switzerland cargo bulletins)
    * src/hw_rhein_feature_pipeline.py    (Hopsworks feature groups)

Why Basel AND Kaub
------------------
A tanker that loads in Rotterdam and sails to Basel has to pass every gauge on
the Rotterdam -> Basel corridor. The load capacity is therefore limited by the
*lowest* water level along the route, not by the level at Basel itself. The
Middle-Rhine bottleneck (Kaub, Rhine-km 546) is routinely the limiting point,
so both stations are collected and a combined "bottleneck" feature is derived.

Verified data sources (checked live 2026-10-17)
----------------------------------------------
1. data.bs.ch, dataset 100089 (Basel-Rheinhalle, 5-minute values)
   CSV export: https://data.bs.ch/api/v2/catalog/datasets/100089/exports/csv
   columns: timestamp (UTC), abfluss [m3/s], pegelhoehe [cm], pegel [m a.s.l.]
   coverage: 2020-06-22 .. now (~664k records). Paging limit: offset+limit
   <= 10000, so long histories are fetched as month windows through the
   ODSQL `where` clause (quoted date literals only).

2. WSV Pegelonline REST API v2
   Basel-Rheinhalle uuid 94f6eff1-4f3f-4850-82e0-a086198e9ffd (number 2310010)
   Kaub            uuid 1d26e504-7f9e-480a-b52c-5932be6549ab (number 25700100)
   parameter W = water level [cm], Q = discharge [m3/s] (Q exists for Kaub,
   NOT for Basel - Basel discharge comes from data.bs.ch).
   Timestamps are published in local time (+01:00/+02:00) and are normalised
   to UTC by the collector.

3. LINDAS / BAFU JSON-LD (latest official Swiss observation only)
   https://environment.ld.admin.ch/foen/hydro/river/observation/2289
   (BAFU station 2289 == Basel-Rheinhalle). Used as a cross-check and for the
   realtime snapshot; timestamps are local time as well.

4. Port of Switzerland (Schweizerische Rheinhaefen) monthly cargo bulletins
   WordPress media API (discovery, JSON):
   https://port-of-switzerland.ch/wp-json/wp/v2/media?search=Bulletin
   PDF: https://port-of-switzerland.ch/wp-content/uploads/YYYY/MM/Bulletin_MM_JJJJ.pdf
   Water-side cargo turnover in tonnes per NST-2007 goods group, including the
   detailed fuel groups 07.21 / 07.22 / 07.23 / 07.25.
   Publication lag: ~4-7 weeks after the reporting month.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Object storage (same bucket/key layout as src/collect_daily_prices.py)
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("RHEIN_S3_BUCKET", "oilprice-features")
RHEIN_LEVELS_KEY = "raw/rhein_levels.csv"
RHEIN_CARGO_KEY = "raw/rhein_cargo.csv"
RHEIN_SNAPSHOT_KEY = "raw/rhein_snapshot.json"
# Bulletins that were already fetched and parsed (including the ones that
# yielded no records), so a daily run does not re-download the whole archive.
RHEIN_BULLETINS_KEY = "raw/rhein_bulletins_parsed.json"


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------
BASEL = {
    "station": "basel",
    "name": "Basel-Rheinhalle",
    "bs_ch_dataset": "100089",
    "wsv_uuid": "94f6eff1-4f3f-4850-82e0-a086198e9ffd",
    "wsv_number": "2310010",
    "bafu_station": "2289",
    "rhine_km": 164.3,
    # data.bs.ch history starts here (verified: sort=timestamp&limit=1)
    "history_start": "2020-06-22",
    # WSV has no discharge series for Basel (Q -> "Timeseries does not exist")
    "wsv_discharge": False,
    "low_water_cm": 350.0,
}

KAUB = {
    "station": "kaub",
    "name": "Kaub (Middle-Rhine bottleneck, Rhine-km 546.23)",
    "bs_ch_dataset": None,
    "wsv_uuid": "1d26e504-7f9e-480a-b52c-5932be6549ab",
    "wsv_number": "25700100",
    "bafu_station": None,
    "rhine_km": 546.23,
    # WSV only serves the last ~30 days for this station
    "history_start": None,
    "wsv_discharge": True,
    "low_water_cm": 30.0,
}

STATIONS = [BASEL, KAUB]

# Shipping-relevant low-water thresholds (not flood thresholds):
#   Basel < 350 cm  -> Basel-bound tankers have to reduce load
#   Kaub  < 30  cm  -> classic Middle-Rhine bottleneck (2018/2022 campaigns)
BASEL_LOW_WATER_CM = BASEL["low_water_cm"]
KAUB_LOW_WATER_CM = KAUB["low_water_cm"]

# ---------------------------------------------------------------------------
# Source endpoints
# ---------------------------------------------------------------------------
BS_CH_CSV_URL = "https://data.bs.ch/api/v2/catalog/datasets/{dataset}/exports/csv"
BS_CH_MAX_LIMIT = 10000  # data.bs.ch: offset + limit must stay <= 10000

WSV_BASE = "https://pegelonline.wsv.de/webservices/rest-api/v2"

LINDAS_OBSERVATION_URL = "https://environment.ld.admin.ch/foen/hydro/river/observation/{station}"

POS_MEDIA_URL = "https://port-of-switzerland.ch/wp-json/wp/v2/media"
POS_BULLETIN_SEARCH = "Bulletin"

# ---------------------------------------------------------------------------
# Collector windows
# ---------------------------------------------------------------------------
DAILY_LOOKBACK_DAYS = 35     # daily run re-fetches this window (late corrections)
BOOTSTRAP_START = BASEL["history_start"]

# ---------------------------------------------------------------------------
# Feature groups (Hopsworks)
# ---------------------------------------------------------------------------
RHEIN_LEVEL_FEATURE_GROUP = "rhein_level"
RHEIN_LEVEL_FEATURE_GROUP_VERSION = 1
RHEIN_LEVEL_FEATURE_VIEW = "rhein_level_daily"
RHEIN_LEVEL_FEATURE_VIEW_VERSION = 1

RHEIN_CARGO_FEATURE_GROUP = "rhein_cargo"
RHEIN_CARGO_FEATURE_GROUP_VERSION = 1
RHEIN_CARGO_FEATURE_VIEW = "rhein_cargo_monthly"
RHEIN_CARGO_FEATURE_VIEW_VERSION = 1

# Daily gauge features (primary key: date). UTC-calendar-day aggregates of the
# raw 5-minute (Basel) / 15-minute (Kaub) series. "*_level_cm" is the daily
# MINIMUM of the gauge height - the value that limits how deep a passing tanker
# may load; the instantaneous value is in raw/rhein_snapshot.json.
RHEIN_LEVEL_FEATURES = [
    "basel_level_cm",
    "basel_level_min_7d",
    "basel_level_mean_7d",
    "basel_level_zscore_30d",
    "basel_low_days_30d",
    "basel_level_trend_7d",
    "basel_discharge_m3s",
    "kaub_level_cm",
    "kaub_level_min_7d",
    "kaub_level_mean_7d",
    "kaub_low_days_30d",
    "kaub_discharge_m3s",
    # min(basel, kaub): the level that limits the reachable load of the tanker
    "bottleneck_level_cm",
    "bottleneck_level_min_7d",
    "rhein_low_water_flag",
]

# Monthly cargo features (primary key: month, event time: published_date).
# Published with a 4-7 week lag -> LAGGED features only, never real-time.
# A z-score of these series is useless (one value per month, strong seasonal
# cycle), so the features are the level plus its month-on-month and
# year-on-year change, which is what carries the "supply is tight" signal.
RHEIN_CARGO_FEATURES = [
    "oil_zufuhr_t",                 # group 07 (NST-2007) / 3 (legacy), Zufuhr
    "oil_abfuhr_t",                 # group 07 / 3, Abfuhr
    "gas_diesel_heizoel_zufuhr_t",  # 07.23 (legacy 325) - the heating-oil volume
    "gas_diesel_heizoel_abfuhr_t",
    "motorbenzin_zufuhr_t",         # 07.21 (legacy 321)
    "jet_petroleum_zufuhr_t",       # 07.22 (legacy 323)
    "lubricants_abfuhr_t",          # 07.25 (legacy 341)
    "oil_zufuhr_prev_month_t",
    "oil_zufuhr_mom_pct",
    "oil_zufuhr_yoy_pct",
    "oil_zufuhr_ytd_t",
    "cargo_total_zufuhr_t",
    "cargo_total_abfuhr_t",
]

# Cargo group codes extracted from the bulletins. The legacy (pre-NST-2007)
# classification has no "07" group: mineral oil is group 3, and its detail
# positions 321/323/325/341 are mapped onto 07.21/07.22/07.23/07.25 by
# src/collect_rhein_cargo.py so one feature definition covers both eras.
CARGO_MAIN_GROUPS = ("07", "3")
CARGO_DETAIL_GROUPS = ("07.21", "07.22", "07.23", "07.25")
CARGO_TOTAL_GROUP = "TOTAL"  # synthetic group for the per-direction table totals

LOCAL_DIR = REPO_ROOT / "data" / "rhein"
LOCAL_LEVELS_CSV = LOCAL_DIR / "rhein_levels.csv"
LOCAL_CARGO_CSV = LOCAL_DIR / "rhein_cargo.csv"
LOCAL_SNAPSHOT_JSON = LOCAL_DIR / "rhein_snapshot.json"

FEATURES_DIR = REPO_ROOT / "data" / "features"
RHEIN_LEVEL_FEATURES_CSV = FEATURES_DIR / "rhein_level_features.csv"
RHEIN_CARGO_FEATURES_CSV = FEATURES_DIR / "rhein_cargo_features.csv"
