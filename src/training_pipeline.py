"""Training pipeline for the oil price prediction model.

Contract (single source of truth: src/model_contract.py)
--------------------------------------------------------
* Features come from `notebooks/04_feature_engineering.ipynb` /
  `src/hw_feature_pipeline.py` via s3://oilprice-features/processed_features.parquet.
* The split is **chronological with a 7-day embargo**: a row at date t carries a
  label for t+7, so the first `TARGET_HORIZON_DAYS` rows after the cut are dropped
  from the test side. The previous `train_test_split(shuffle=True)` leaked the
  future into training and understated MSE by a factor of ~330.
* `seasonal_index` is **not** taken from the parquet. The parquet value averages
  month/year ratios over *all* years, so a 2010 row carries a 2022-informed
  coefficient. `ExpandingSeasonalIndex` (src/seasonal_index.py) recomputes it
  inside the pipeline from strictly-past years only.
* The target is the **stationary delta** `target_week_change`
  (= target_next_week_price - current_price). RandomForest cannot extrapolate the
  price level: 67.7% of hold-out targets lie above the maximum training target.
  Predictions are mapped back to the level for reporting.
* `season` stays a raw string and is one-hot encoded inside the pipeline. The old
  integer-coding produced a position-keyed model that silently degraded whenever
  the serving side encoded categories differently.
* The model is exported only if it beats the persistence baseline.
* S3 credentials are read from the environment (secretKeyRef "s3-credentials" in
  k8s/training-cronjob.yaml). The pipeline never shells out to kubectl.

Run:
    python src/training_pipeline.py
"""

import json
import os
import sys

import joblib
import mlflow
import pandas as pd
import s3fs
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model_contract import (
    CATEGORICAL_FEATURES,
    DELTA_TARGET,
    LEVEL_TARGET,
    MAX_SKILL_VS_PERSISTENCE,
    TARGET_HORIZON_DAYS,
    ContractError,
    ModelContract,
    chronological_split,
    persistence_baselines,
    validate_features,
    volatility_buckets,
)
from s3_credentials import get_s3_options  # re-exported: hw_*_training_pipeline imports it from here
from seasonal_index import ExpandingSeasonalIndex

FEATURES_S3_PATH = "s3://oilprice-features/processed_features.parquet"
MODEL_S3_PATH = "s3://oilprice-models/oil_price_model.pkl"
CONTRACT_S3_PATH = "s3://oilprice-models/model_features.json"
MLFLOW_EXPERIMENT = "oilprice prediction"
# Same tracking server as notebook 06 ("Set up MLflow Experiment" cell)
MLFLOW_TRACKING_URI_DEFAULT = "http://mlflow.mlflow.svc.cluster.local"

# Must match AGG_FEATURES + RT_FEATURES in 04_feature_engineering.ipynb
AGG_FEATURES = [
    "price_mean_7d", "price_std_30d",
    "demand_mean_7d", "demand_mean_30d",
    "month", "season", "month_sin", "month_cos", "seasonal_index", "price_mean_dow_4w",
]
RT_FEATURES = ["current_price", "current_demand"]
FEATURES = AGG_FEATURES + RT_FEATURES
# `date` is a model *input*, not a feature: ExpandingSeasonalIndex needs it to keep
# the seasonal index point-in-time. It is never passed on to the regressor.
MODEL_INPUTS = FEATURES + ["date"]

N_ESTIMATORS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42



# get_s3_options() lives in src/s3_credentials.py and is re-exported above: the
# credentials arrive as env vars from secretKeyRef "s3-credentials", so the Job pod
# never needs kubectl nor RBAC to read Secrets in the rook-ceph namespace.


def load_features(s3_options: dict) -> pd.DataFrame:
    """Load the processed-features parquet written to S3 by 04_feature_engineering.ipynb."""
    s3 = s3fs.S3FileSystem(**s3_options)
    if not s3.exists(FEATURES_S3_PATH):
        raise FileNotFoundError(
            f"Processed features not found at {FEATURES_S3_PATH}. "
            "Run notebooks/04_feature_engineering.ipynb first."
        )
    df = pd.read_parquet(FEATURES_S3_PATH, storage_options=s3_options)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {FEATURES_S3_PATH}")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically, enforce the contract, and materialise the delta target.

    Returns the clean frame (features + `date` + both targets). The split helper
    derives X and y from it so the embargo is applied to *post-dropna* positions,
    which is the only version that is actually leak-free.
    """
    if "date" not in df.columns:
        raise ContractError("'date' column required for the chronological split")

    df = df.sort_values("date").reset_index(drop=True)

    required = MODEL_INPUTS + [LEVEL_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(f"Missing columns in dataset: {missing}")

    # `season` must stay a raw string: OneHotEncoder is fit on the training
    # vocabulary inside the pipeline. Integer coding here would defeat that.
    if not pd.api.types.is_string_dtype(df["season"]) and not pd.api.types.is_object_dtype(df["season"]):
        raise ContractError(
            f"'season' must be a string column, got dtype {df['season'].dtype}"
        )

    if DELTA_TARGET not in df.columns:
        df[DELTA_TARGET] = df[LEVEL_TARGET] - df["current_price"]

    before = len(df)
    df = df.dropna(subset=MODEL_INPUTS + [LEVEL_TARGET, DELTA_TARGET])
    print(f"Rows kept: {len(df)} (dropped {before - len(df)} with missing values)")
    if before - len(df):
        print("Missing values per column:")
        print(df.isnull().sum().to_string())

    validate_features(df[MODEL_INPUTS], ModelContract(features=MODEL_INPUTS))
    return df


def build_model(X_train: pd.DataFrame) -> Pipeline:
    """ExpandingSeasonalIndex -> ColumnTransformer -> RandomForestRegressor.

    Step order matters. `ExpandingSeasonalIndex` runs first and *overwrites* the
    leaky `seasonal_index` column that arrived in the frame, so the
    ColumnTransformer's `numeric_cols` - derived from the frame BEFORE the
    transformer runs - already contains it. Appending it again would select the
    column twice and sklearn aborts with
    `DuplicateError: Expected unique column names`.

    The categorical vocabulary and the seasonal maps are fit on X_train only and
    travel inside the artifact, so serving reuses exactly the training encoding.
    """
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    missing_cats = set(CATEGORICAL_FEATURES) - set(categorical_cols)
    if missing_cats:
        raise ContractError(
            f"Categorical features missing from training frame: {sorted(missing_cats)}"
        )
    for col in categorical_cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            raise ContractError(
                f"'{col}' is numeric in the training frame; raw category strings are "
                f"required so OneHotEncoder is vocabulary-keyed, not position-keyed"
            )

    numeric_cols = [
        c for c in X_train.columns
        if c not in categorical_cols and pd.api.types.is_numeric_dtype(X_train[c])
    ]
    # `date` is consumed by ExpandingSeasonalIndex and must NOT reach the
    # passthrough block: RandomForest converts X strictly to float and aborts on
    # Timestamp with "float() argument must be a string or a real number".
    dropped = [c for c in X_train.columns if c not in categorical_cols and c not in numeric_cols]
    if dropped:
        print(f"non-numeric inputs consumed by feature_engineering only: {dropped}")

    if "seasonal_index" not in numeric_cols:
        raise ContractError(
            "'seasonal_index' is not a numeric column in the training frame; "
            "ExpandingSeasonalIndex writes it and the passthrough block must select it"
        )

    print(f"Pipeline setup -> categorical: {categorical_cols} | numeric: {numeric_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("numeric", "passthrough", numeric_cols),
        ]
    )
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                ExpandingSeasonalIndex(
                    date_col="date", price_col="price_mean_7d", out_col="seasonal_index"
                ),
            ),
            ("preprocessing", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, train: pd.DataFrame, test: pd.DataFrame,
             input_columns: list[str] | None = None) -> dict:
    """Score the delta model on the price level it will actually be judged on.

    The model predicts `target_week_change`. Serving renders a price, so the
    reported metrics reconstruct the level as current_price + predicted_delta and
    compare it against persistence. Reporting delta-MSE alone would hide that the
    model is worse than 'tomorrow is today'.
    """
    cols = input_columns or MODEL_INPUTS
    predicted_delta = model.predict(test[cols])
    truth_level = test[LEVEL_TARGET].to_numpy(dtype="float64")
    level_pred = test["current_price"].to_numpy(dtype="float64") + predicted_delta

    level_mse = float(mean_squared_error(truth_level, level_pred))
    level_r2 = float(r2_score(truth_level, level_pred))
    delta_mse = float(mean_squared_error(test[DELTA_TARGET], predicted_delta))
    delta_r2 = float(r2_score(test[DELTA_TARGET], predicted_delta))

    baselines = persistence_baselines(test, LEVEL_TARGET)
    reference = baselines.get("current_price", float("nan"))
    skill = level_mse / reference if reference else float("inf")

    # The single skill number is the average over weeks that moved and weeks
    # that did not. Bucketing the same hold-out shows which of the two the model
    # is actually failing on - see model_contract.volatility_buckets.
    buckets = volatility_buckets(test, level_pred)

    # Train-side sanity: the model must at least fit the window it was trained on.
    train_delta = model.predict(train[cols])
    train_level = train["current_price"].to_numpy(dtype="float64") + train_delta
    train_level_mse = float(mean_squared_error(train[LEVEL_TARGET], train_level))

    return {
        "level_mse": level_mse,
        "level_r2": level_r2,
        "delta_mse": delta_mse,
        "delta_r2": delta_r2,
        "train_level_mse": train_level_mse,
        "baselines": baselines,
        "skill_vs_persistence": skill,
        "buckets": buckets,
    }



def save_model_to_s3(model: Pipeline, s3_options: dict) -> None:
    s3 = s3fs.S3FileSystem(**s3_options)
    with s3.open(MODEL_S3_PATH, "wb") as f:
        joblib.dump(model, f)
    print(f"Model saved to {MODEL_S3_PATH}")


def save_contract_to_s3(contract: ModelContract, s3_options: dict) -> None:
    """Write the sidecar the inference app needs to rebuild the exact frame."""
    s3 = s3fs.S3FileSystem(**s3_options)
    with s3.open(CONTRACT_S3_PATH, "wb") as f:
        f.write(json.dumps(contract.to_dict(), indent=2, ensure_ascii=False).encode("utf-8"))
    print(f"Feature contract saved to {CONTRACT_S3_PATH}")


def main() -> int:
    try:
        s3_options = get_s3_options()
        df = load_features(s3_options)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    try:
        frame = prepare_data(df)
        train, test, audit = chronological_split(
            frame, test_size=TEST_SIZE, horizon_days=TARGET_HORIZON_DAYS
        )
    except ContractError as e:
        print(f"Contract violation: {e}", file=sys.stderr)
        return 1

    X_train, y_train = train[MODEL_INPUTS], train[DELTA_TARGET]
    print(f"X_train: {X_train.shape} | y_train (delta): {y_train.shape}")
    print(
        f"train {audit['train_start_date']} .. {audit['train_end_date']} "
        f"(labels reach {audit['train_label_end_date']}) | "
        f"test {audit['test_start_date']} .. {audit['test_end_date']} | "
        f"embargo {audit['embargo_days']}d, {audit['dropped_by_embargo']} rows dropped"
    )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI_DEFAULT
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # Autolog must be active BEFORE model.fit(): it logs the model exactly once
    # (name "model") together with the training-dataset association.
    mlflow.autolog()

    model = build_model(X_train)
    with mlflow.start_run():
        model.fit(X_train, y_train)
        metrics = evaluate(model, train, test)

        contract = ModelContract(
            features=MODEL_INPUTS,
            target=DELTA_TARGET,
            level_target=LEVEL_TARGET,
            horizon_days=TARGET_HORIZON_DAYS,
            extra={
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "seasonal_index_source": "pipeline (ExpandingSeasonalIndex)",
                "prediction_rule": "price = current_price + predicted target_week_change",
            },
        )

        for key, value in audit.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_type", "Pipeline(ExpandingSeasonalIndex + ColumnTransformer + RandomForestRegressor)")
        mlflow.log_param("target", DELTA_TARGET)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model_inputs", ",".join(contract.features))

        mlflow.log_metric("level_mse", metrics["level_mse"])
        mlflow.log_metric("level_r2", metrics["level_r2"])
        mlflow.log_metric("delta_mse", metrics["delta_mse"])
        mlflow.log_metric("delta_r2", metrics["delta_r2"])
        mlflow.log_metric("train_level_mse", metrics["train_level_mse"])
        for name, value in metrics["baselines"].items():
            mlflow.log_metric(f"baseline_level_mse_{name}", value)
        mlflow.log_metric("skill_vs_persistence", metrics["skill_vs_persistence"])

        gate_ok = metrics["skill_vs_persistence"] < MAX_SKILL_VS_PERSISTENCE
        mlflow.log_metric("promotion_gate", 1 if gate_ok else 0)

    print(f"Level MSE: {metrics['level_mse']:.2f} (R2 {metrics['level_r2']:.3f})")
    print(f"Delta MSE: {metrics['delta_mse']:.3f} (R2 {metrics['delta_r2']:.3f})")
    for name, value in metrics["baselines"].items():
        print(f"  baseline {name}: {value:.2f}")
    print(f"Skill vs persistence: {metrics['skill_vs_persistence']:.3f} "
          f"(< {MAX_SKILL_VS_PERSISTENCE} required)")

    if not gate_ok:
        print(
            "PROMOTION BLOCKED: the model is not better than persistence on the "
            "chronological hold-out. Run logged, artifact NOT exported.",
            file=sys.stderr,
        )
        return 2

    save_model_to_s3(model, s3_options)
    save_contract_to_s3(contract, s3_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

