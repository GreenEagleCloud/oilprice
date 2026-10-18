"""Training pipeline with Hopsworks Feature Store + Model Registry (TODO.md item 7).

Trains the same RandomForest pipeline as `src/training_pipeline.py`, but:
  * the training data comes from the Hopsworks feature view `heizoel24_weekly`
    (feature store instead of the S3 parquet),
  * the model is additionally registered in the Hopsworks Model Registry
    (MLflow logging unchanged, optional S3 export for the legacy inference app).

Run:
    python src/hw_training_pipeline.py
"""

import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error

from hw_config import (
    FEATURES,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI_DEFAULT,
    MODEL_FILE_NAME,
    MODEL_NAME,
    N_ESTIMATORS,
    RANDOM_STATE,
    TEST_SIZE,
    login,
    weekly_model_features,
)
from model_artifact import write_model_features
from training_pipeline import build_model, get_s3_options, save_model_to_s3


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Non-numeric feature columns -> integer codes.

    Same as `prepare_data` in src/training_pipeline.py: RandomForestRegressor
    (sklearn 1.9.1) has no native categorical support and converts X strictly
    to float, so string columns (e.g. "season") must be encoded first.
    """
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes
    return X


def register_model(project, model, mse: float, features: list | None = None):
    """Register the model in the Hopsworks Model Registry.

    Hopsworks 5.x model registry: `Model.save(model_folder)` uploads a folder
    of model files as the model artifact. With `id=None` a new model is
    created, and with `version=None` the next version number is assigned
    automatically (1, 2, 3, ...).

    `features` is written as a sidecar file so the inference app knows exactly
    which feature-view columns this model expects (see src/model_artifact.py).
    """
    from hsml.model import Model

    with tempfile.TemporaryDirectory(prefix="hw_model_") as tmp:
        model_dir = Path(tmp) / MODEL_NAME
        model_dir.mkdir()
        joblib.dump(model, model_dir / MODEL_FILE_NAME)
        if features:
            write_model_features(model_dir, MODEL_NAME, features)

        registry = project.get_model_registry()
        reg_model = Model(
            id=None,
            name=MODEL_NAME,
            description="RandomForest pipeline -> target_next_week_price (heizoel24, weekly)",
            framework="SKLEARN",
            metrics={"mse": float(mse)},
            # The default model registry id equals the project id; it must be
            # passed explicitly, otherwise Model.save() builds the REST path
            # with registry id None and the backend returns 404.
            model_registry_id=registry.model_registry_id,
        )
        return reg_model.save(str(model_dir))


def main() -> int:
    # 1) Training data from the Hopsworks feature view
    project = login()
    fs = project.get_feature_store()
    fv = fs.get_feature_view(FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=TEST_SIZE)

    # The feature group also holds the most recent dates whose 7-day-ahead
    # label is NULL (stored for serving, see hw_feature_pipeline.py); rows
    # without a label are useless for training and are dropped here.
    def _drop_unlabeled(X: pd.DataFrame, y) -> tuple:
        y = pd.DataFrame(y)
        keep = y.iloc[:, 0].notna().to_numpy()
        return X[keep], y[keep]

    n_total = len(y_train) + len(y_test)
    X_train, y_train = _drop_unlabeled(X_train, y_train)
    X_test, y_test = _drop_unlabeled(X_test, y_test)
    print(f"train_test_split: {n_total} rows, dropped {n_total - len(y_train) - len(y_test)} rows without a label")

    # 2) Model feature columns: the config says which drivers should be used,
    #    the feature view says what is actually available. Driver columns that
    #    the view does not have (yet) are reported and skipped instead of
    #    failing the training.
    wanted = weekly_model_features()
    missing = [col for col in wanted if col not in X_train.columns]
    if missing:
        print(f"WARNING: feature view '{FEATURE_VIEW_NAME}' v{FEATURE_VIEW_VERSION} lacks {missing}")
    model_features = [col for col in wanted if col in X_train.columns]
    driver_features = [col for col in model_features if col not in FEATURES]
    X_train, X_test = X_train[model_features], X_test[model_features]

    X_train, X_test = encode_features(X_train), encode_features(X_test)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"driver features used: {len(driver_features)} ({', '.join(driver_features)})")

    # 2) Train + MLflow (same run setup as src/training_pipeline.py)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI_DEFAULT
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # Autolog must be active BEFORE model.fit(): it logs the model exactly once
    # (name "model") together with the training metrics.
    mlflow.autolog()

    model = build_model(X_train)
    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("model_type", "Pipeline(ColumnTransformer + RandomForestRegressor)")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("data_source", f"hopsworks feature view {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}")
        mlflow.log_param("n_features", len(model_features))
        mlflow.log_param("driver_features", ",".join(driver_features) or "none")
        mlflow.log_metric("mse", mse)
    print(f"MSE: {mse}")

    # 3) Register in the Hopsworks Model Registry (auto versioning)
    registered = register_model(project, model, mse, model_features)
    print(f"Registered '{MODEL_NAME}' (version {registered.version}) in the Hopsworks Model Registry")

    # 4) Optional: export to S3 for the legacy inference app
    #    (skipped gracefully when no cluster access is available)
    try:
        s3_options = get_s3_options()
        save_model_to_s3(model, s3_options)
    except Exception as e:
        print(f"Note: S3 export skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
