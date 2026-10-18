"""Training pipeline for the forward oil-price forecast model.

This trains the model for the `heizoel24_forward_forecast` feature view.

The model predicts:

    target_price_week_after_target_date

from features known at `origin_date`, including:

    forecast_horizon_days = target_date - origin_date

This is the model used when the requested prediction date is in the future
and therefore does not exist in the normal weekly feature view.
"""

import os
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from hsml.model import Model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from hw_config import (
    FORWARD_FEATURE_VIEW_NAME,
    FORWARD_FEATURE_VIEW_VERSION,
    FORWARD_FEATURES,
    FORWARD_MAX_DEPTH,
    FORWARD_MAX_FEATURES,
    FORWARD_MIN_SAMPLES_LEAF,
    FORWARD_MODEL_FILE_NAME,
    FORWARD_MODEL_NAME,
    FORWARD_N_ESTIMATORS,
    FORWARD_PRIMARY_KEYS,
    FORWARD_TARGET,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI_DEFAULT,
    MODEL_BUCKET,
    RANDOM_STATE,
    TEST_SIZE,
    forward_model_features,
    login,
)
from model_artifact import write_model_features
from training_pipeline import get_s3_options


def _drop_unlabeled(X: pd.DataFrame, y):
    if isinstance(y, pd.DataFrame):
        mask = y.iloc[:, 0].notna().to_numpy()
        return X[mask], y[mask]
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def _encode_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes
    return X


def build_forward_model(X_train: pd.DataFrame):
    """Build a compact RandomForest pipeline for the forward feature view.

    The forward training set contains many origin_date/target_date combinations,
    so an unconstrained RandomForest can become unnecessarily large. The
    constrained forest keeps the registered model small enough for inference
    while still using forecast_horizon_days as a normal numeric feature.
    """
    categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns
    numeric_cols = X_train.columns.difference(categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )
    return make_pipeline(
        preprocessor,
        RandomForestRegressor(
            n_estimators=FORWARD_N_ESTIMATORS,
            max_depth=FORWARD_MAX_DEPTH,
            min_samples_leaf=FORWARD_MIN_SAMPLES_LEAF,
            max_features=FORWARD_MAX_FEATURES,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )


def register_forward_model(project, model, mse: float, features: list | None = None):
    """Register the forward model in the Hopsworks Model Registry."""
    registry = project.get_model_registry()

    with tempfile.TemporaryDirectory(prefix="hw_forward_model_") as tmp:
        model_dir = Path(tmp) / FORWARD_MODEL_NAME
        model_dir.mkdir()
        joblib.dump(model, model_dir / FORWARD_MODEL_FILE_NAME)
        if features:
            write_model_features(model_dir, FORWARD_MODEL_NAME, features)

        reg_model = Model(
            id=None,
            name=FORWARD_MODEL_NAME,
            description=(
                "RandomForest pipeline -> target_price_week_after_target_date "
                "(heizoel24 forward forecast with forecast_horizon_days)"
            ),
            framework="SKLEARN",
            metrics={"mse": float(mse)},
            model_registry_id=registry.model_registry_id,
        )
        return reg_model.save(str(model_dir))


def save_forward_model_to_s3(model, s3_options: dict) -> None:
    import s3fs

    s3 = s3fs.S3FileSystem(**s3_options)
    path = f"s3://{MODEL_BUCKET}/{FORWARD_MODEL_FILE_NAME}"
    with s3.open(path, "wb") as f:
        joblib.dump(model, f)
    print(f"Forward model saved to {path}")


def main() -> int:
    try:
        project = login()
        fs = project.get_feature_store()
        fv = fs.get_feature_view(
            FORWARD_FEATURE_VIEW_NAME, version=FORWARD_FEATURE_VIEW_VERSION
        )

        X_train, X_test, y_train, y_test = fv.train_test_split(test_size=TEST_SIZE)

        # Primary keys are used for online lookup. They must not become model
        # features; forecast_horizon_days is a normal feature and is kept.
        for col in FORWARD_PRIMARY_KEYS:
            if col in X_train.columns:
                X_train = X_train.drop(columns=[col])
            if col in X_test.columns:
                X_test = X_test.drop(columns=[col])

        missing_features = [col for col in FORWARD_FEATURES if col not in X_train.columns]
        if missing_features:
            raise ValueError(f"Forward feature view is missing features: {missing_features}")

        # Driver columns are optional: a forward view created before the Rhine /
        # market join simply lacks them, and the model is then trained without.
        wanted = forward_model_features()
        missing_drivers = [col for col in wanted if col not in X_train.columns]
        if missing_drivers:
            print(f"WARNING: forward feature view lacks driver features: {missing_drivers}")
        model_features = [col for col in wanted if col in X_train.columns]
        driver_features = [col for col in model_features if col not in FORWARD_FEATURES]

        X_train = X_train[model_features]
        X_test = X_test[model_features]
        print(f"driver features used: {len(driver_features)} ({', '.join(driver_features)})")

        n_total = len(y_train) + len(y_test)
        X_train, y_train = _drop_unlabeled(X_train, y_train)
        X_test, y_test = _drop_unlabeled(X_test, y_test)
        print(
            f"train_test_split: {n_total} rows, dropped "
            f"{n_total - len(y_train) - len(y_test)} rows without a label"
        )

        X_train, X_test = _encode_features(X_train), _encode_features(X_test)

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI_DEFAULT
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.autolog()

        model = build_forward_model(X_train)
        with mlflow.start_run():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            y_test_series = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test
            mse = mean_squared_error(y_test_series, predictions)

            mlflow.log_param("model_type", "Pipeline(ColumnTransformer + RandomForestRegressor)")
            mlflow.log_param("model_variant", "forward_forecast")
            mlflow.log_param("n_estimators", FORWARD_N_ESTIMATORS)
            mlflow.log_param("max_depth", FORWARD_MAX_DEPTH)
            mlflow.log_param("min_samples_leaf", FORWARD_MIN_SAMPLES_LEAF)
            mlflow.log_param("max_features", FORWARD_MAX_FEATURES)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_features", len(model_features))
            mlflow.log_param("driver_features", ",".join(driver_features) or "none")
            mlflow.log_metric("mse", mse)

        registered = register_forward_model(project, model, mse, model_features)
        print(
            f"Registered '{FORWARD_MODEL_NAME}' "
            f"(version {registered.version}) in the Hopsworks Model Registry"
        )

        try:
            s3_options = get_s3_options()
            save_forward_model_to_s3(model, s3_options)
        except Exception as e:
            print(f"Note: S3 export skipped: {e}")

        print(f"MSE: {mse}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
