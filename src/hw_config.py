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

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths & names (kept in sync with the notebooks and src/training_pipeline.py)
# ---------------------------------------------------------------------------
DATA_FILE = REPO_ROOT / "data" / "heizoel24_heizoel_daily_ch_full.csv"
FEATURES_LOCAL_CSV = REPO_ROOT / "data" / "features" / "heizoel24_features.csv"

FEATURE_GROUP_NAME = "heizoel24"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "heizoel24_weekly"
FEATURE_VIEW_VERSION = 1

FORWARD_FEATURE_GROUP_NAME = "heizoel24_forward"
FORWARD_FEATURE_GROUP_VERSION = 1
FORWARD_FEATURE_VIEW_NAME = "heizoel24_forward_forecast"
FORWARD_FEATURE_VIEW_VERSION = 1

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
