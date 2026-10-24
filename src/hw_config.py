"""Shared configuration for the Hopsworks pipelines (Feature Store + Model Registry).

Used by:
    * src/hw_feature_pipeline.py            (TODO.md item 6)
    * src/hw_forward_feature_pipeline.py    (forward forecast feature view)
    * src/hw_training_pipeline.py           (TODO.md item 7)
    * src/hw_forward_training_pipeline.py   (forward forecast model)
    * src/inference/hw_app.py               (TODO.md item 8)

Credentials are read from environment variables (a local `.env` file is loaded
automatically, see `.env.example`). The variable names are the standard
Hopsworks SDK variables:

    HOPSWORKS_HOST      e.g. eu-west.cloud.hopsworks.ai
    HOPSWORKS_PORT      default 443
    HOPSWORKS_PROJECT   project name
    HOPSWORKS_API_KEY   API key (Hopsworks UI > Account Settings > API Keys)
    HOPSWORKS_ENGINE    "python" (local execution, default) or "spark"
    MLFLOW_TRACKING_URI optional override (default: in-cluster MLflow server)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from market_config import FEATURES_LOCAL_CSV as MARKET_FEATURES_CSV
from market_config import (
    MARKET_FEATURES,
    MARKET_FEATURE_GROUP,
    MARKET_FEATURE_GROUP_VERSION,
    MARKET_FEATURE_VIEW,
    MARKET_FEATURE_VIEW_VERSION,
)
from rhein_config import (
    RHEIN_LEVEL_FEATURES,
    RHEIN_LEVEL_FEATURE_VIEW,
    RHEIN_LEVEL_FEATURE_VIEW_VERSION,
    RHEIN_LEVEL_FEATURES_CSV,
)
# Re-exported from src/model_contract.py so the forward target names live in one
# place: the feature pipelines, the training pipelines and the inference app all
# read the same constants instead of repeating string literals.
from model_contract import (  # noqa: F401
    FORWARD_DELTA_TARGET,
    FORWARD_ORIGIN_PRICE,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths & names (kept in sync with the notebooks and src/training_pipeline.py)
# ---------------------------------------------------------------------------
DATA_FILE = REPO_ROOT / "data" / "heizoel24_heizoel_daily_ch_full.csv"
FEATURES_LOCAL_CSV = REPO_ROOT / "data" / "features" / "heizoel24_features.csv"

FEATURE_GROUP_NAME = "heizoel24"
# v2 = v1 columns + the joined Rhine/market driver columns. v1 (price features
# only) is kept so the previously served model stays reproducible.
FEATURE_GROUP_VERSION = 2
FEATURE_VIEW_NAME = "heizoel24_weekly"
# v3 = v2 features + the `date` anchor. Hopsworks does not expose a feature
# group's primary key in a feature view unless it is selected explicitly, and the
# training pipeline cannot build a chronological split - or keep the seasonal
# index point-in-time - without it. v1/v2 stay untouched so the models already
# served from them remain reproducible.
FEATURE_VIEW_VERSION = 3

FORWARD_FEATURE_GROUP_NAME = "heizoel24_forward"
FORWARD_FEATURE_GROUP_VERSION = 2
FORWARD_FEATURE_VIEW_NAME = "heizoel24_forward_forecast"
# v3 = v2 features + the `origin_date` anchor. The forward feature group is keyed
# by (target_date, origin_date) and the forward split is chronological over
# origin_date, so the view has to expose it. `target_date` is deliberately NOT
# selected: origin_date + forecast_horizon_days already determine it, so exposing
# it would only add a leak risk.
FORWARD_FEATURE_VIEW_VERSION = 3

# Anchor columns the feature views must expose (see the version notes above).
# They are primary keys of the feature group and split anchors, never model
# inputs: weekly_model_features() / forward_model_features() do not list them.
DATE_COLUMN = "date"
FORWARD_ORIGIN_DATE_COLUMN = "origin_date"

MODEL_NAME = "oil_price_model"
MODEL_FILE_NAME = "oil_price_model.pkl"  # joblib file inside the registry model folder
MODEL_BUCKET = "oilprice-models"         # S3 fallback bucket (see src/training_pipeline.py)

FORWARD_MODEL_NAME = "oil_price_forward_model"
FORWARD_MODEL_FILE_NAME = "oil_price_forward_model.pkl"
FORWARD_MODEL_S3_KEY = "oil_price_forward_model.pkl"

# Must match AGG_FEATURES + RT_FEATURES in 04_feature_engineering.ipynb
AGG_FEATURES = [
    "price_mean_7d", "price_std_30d",
    "demand_mean_7d", "demand_mean_30d",
    "month", "season", "month_sin", "month_cos", "seasonal_index", "price_mean_dow_4w",
]
RT_FEATURES = ["current_price", "current_demand"]
FEATURES = AGG_FEATURES + RT_FEATURES

# ---------------------------------------------------------------------------
# External drivers joined into the training feature views
# ---------------------------------------------------------------------------
# Rhine gauge (src/hw_rhein_feature_pipeline.py) and oil market (src/
# hw_oil_market_feature_pipeline.py) features are LEFT-joined onto the price
# rows by date, so a price row is never dropped when a driver is missing
# (Rhine starts 2020-06-22, the market series start 2003/2007).
#
# Only a compact, datum-consistent subset of the 15 gauge features is used:
#   basel_level_cm             daily minimum at Basel-Rheinhalle (shipping day)
#   basel_level_min_7d         worst level of the shipping week
#   basel_level_zscore_30d     how unusual the current level is
#   basel_low_days_30d         duration of the low-water phase
#   basel_level_trend_7d       is the situation easing or tightening
#   rhein_low_water_flag       binary alert (each station vs its own threshold)
#
# bottleneck_level_cm / _min_7d are deliberately NOT model features: Basel and
# Kaub heights are measured against different gauge datums (Basel ~470 cm while
# Kaub runs from about -6 cm to 30 cm), so min(Basel, Kaub) is not a comparable
# depth - it would make the feature jump by two orders of magnitude the day Kaub
# coverage starts. The columns stay in the feature group for the /rhein snapshot.
#
# The per-station Kaub columns are not used either: WSV publishes Kaub only for
# the last ~30 days, so they would be NULL for almost the whole training history
# (raw/rhein_kaub_raw.csv accumulates the history for future model versions).
RHEIN_MODEL_FEATURES = [
    "basel_level_cm",
    "basel_level_min_7d",
    "basel_level_zscore_30d",
    "basel_low_days_30d",
    "basel_level_trend_7d",
    "rhein_low_water_flag",
]

# Market drivers (ICE Gasoil USD/t, USD/CHF and the derived CHF/100l reference
# price). gasoil_chf_tonne is the exact product of the other two columns and is
# therefore left out; the 7-day mean/change carry the momentum.
MARKET_MODEL_FEATURES = [
    "market_price_chf_100l",
    "market_price_chf_100l_mean_7d",
    "market_price_chf_100l_chg_7d_pct",
    "gasoil_usd_tonne",
    "usd_chf",
]


def _flag(name: str) -> bool:
    return os.getenv(name, "1").strip().lower() not in ("0", "false", "no", "off")


# Both joins can be switched off per environment variable, e.g. to retrain the
# baseline model for an A/B comparison without redeploying.
USE_RHEIN_FEATURES = _flag("OILPRICE_USE_RHEIN_FEATURES")
USE_MARKET_FEATURES = _flag("OILPRICE_USE_MARKET_FEATURES")


def weekly_model_features() -> list:
    """Feature columns of the weekly view (price features + enabled drivers)."""
    columns = list(FEATURES)
    if USE_RHEIN_FEATURES:
        columns += RHEIN_MODEL_FEATURES
    if USE_MARKET_FEATURES:
        columns += MARKET_MODEL_FEATURES
    return columns


def forward_model_features() -> list:
    """Feature columns of the forward view (origin features + enabled drivers)."""
    columns = list(FORWARD_FEATURES)
    if USE_RHEIN_FEATURES:
        columns += FORWARD_RHEIN_FEATURES
    if USE_MARKET_FEATURES:
        columns += FORWARD_MARKET_FEATURES
    return columns


# Notebook-04 labels. Only TARGET is used for training; the others stay in the
# feature group (documentation/analysis) but are NOT selected by the feature
# view, so they can never leak into the training features.
TARGET = "target_next_week_price"

# Forward-forecast feature view:
#
# The weekly feature view above answers:
#   "Given date t, what is the average price for t+1..t+7?"
#
# The forward feature view answers:
#   "Given the latest known origin_date o and a future target_date t,
#    what is the average price for t+1..t+7?"
#
# This makes predictions possible for dates that do not yet have a weekly
# feature row, e.g. 2026-11-11 when the feature store only contains data up
# to 2026-10-09.
FORWARD_PRIMARY_KEYS = ["target_date", "origin_date"]
FORWARD_EVENT_TIME = "origin_date"

FORWARD_FEATURES = [
    "origin_price_mean_7d",
    "origin_price_std_30d",
    "origin_demand_mean_7d",
    "origin_demand_mean_30d",
    "origin_price_mean_dow_4w",
    "origin_current_price",
    "origin_current_demand",
    "target_month",
    "target_season",
    "target_month_sin",
    "target_month_cos",
    "target_seasonal_index",
    "forecast_horizon_days",
]

# Driver features of the origin date (same values as the weekly view, but named
# with the origin_ prefix because a forward row is keyed by target_date +
# origin_date and every driver must be known at origin_date).
FORWARD_RHEIN_FEATURES = [f"origin_{name}" for name in RHEIN_MODEL_FEATURES]
FORWARD_MARKET_FEATURES = [f"origin_{name}" for name in MARKET_MODEL_FEATURES]

# Level label stored in the forward feature group. The forward model does NOT
# regress it: it regresses FORWARD_DELTA_TARGET = FORWARD_TARGET -
# origin_current_price, exactly like the weekly model regresses
# target_week_change instead of target_next_week_price. The delta is derived at
# training time and is deliberately not a column of the feature view - a stored
# delta column would be a label-shaped feature the view could hand to the model.
FORWARD_TARGET = "target_price_week_after_target_date"

# Training horizons used to teach the model how predictions change with the
# distance between origin_date and target_date.
#
# Serving horizons are kept dense so any requested future date up to 90 days
# can be looked up online. Training horizons are coarser to keep the feature
# group smaller; RandomForest can still use the exact numeric horizon value at
# inference time.
FORWARD_SERVING_HORIZONS = list(range(1, 91))
FORWARD_TRAINING_HORIZONS = (
    list(range(1, 31))
    + [33, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
)
FORWARD_HORIZONS = sorted(set(FORWARD_TRAINING_HORIZONS + FORWARD_SERVING_HORIZONS))
LABELS = ["target_next_week_price", "target_week_change", "target_trend", "target_next_month_price"]

N_ESTIMATORS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Forward models are trained on a much larger feature group (origin_date x
# target_date combinations). Keep the forward RandomForest deliberately small
# so the registered artifact stays practical for online inference.
FORWARD_N_ESTIMATORS = 50
FORWARD_MAX_DEPTH = 12
FORWARD_MIN_SAMPLES_LEAF = 20
FORWARD_MAX_FEATURES = 0.5

MLFLOW_EXPERIMENT = "oilprice prediction"
MLFLOW_TRACKING_URI_DEFAULT = "http://mlflow.mlflow.svc.cluster.local"


def login():
    """Log in to Hopsworks using the env vars above and return the project."""
    import hopsworks

    host = os.environ.get("HOPSWORKS_HOST")
    project = os.environ.get("HOPSWORKS_PROJECT")
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    missing = [
        name
        for name, value in (
            ("HOPSWORKS_HOST", host),
            ("HOPSWORKS_PROJECT", project),
            ("HOPSWORKS_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise EnvironmentError(
            f"Missing Hopsworks environment variables: {', '.join(missing)}. "
            "Set them in the environment or in .env (see .env.example)."
        )

    return hopsworks.login(
        host=host,
        port=int(os.environ.get("HOPSWORKS_PORT", "443")),
        project=project,
        api_key_value=api_key,
        engine=os.environ.get("HOPSWORKS_ENGINE", "python"),
    )
