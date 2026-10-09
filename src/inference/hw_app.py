"""Hopsworks inference app (Flask, TODO.md item 8).

POST /predict
    {
        "date": "2026-09-25",          # optional; default: latest day in the feature store
        "current_price": 164.89,       # optional; live data / user input
        "current_demand": 12.28        # optional; live data / user input
    }

1. Live data: optional `current_price` / `current_demand` from the request
   body (real-time features, TODO.md 8.1) - if not provided, the stored
   online values for the given date are used.
2. Feature vector: fetched from the Hopsworks online feature store via the
   feature view `heizoel24_weekly`.
3. Model: loaded from the Hopsworks Model Registry (latest version of
   `oil_price_model`); falls back to S3 (s3://oilprice-models/) when the
   registry is not reachable.
4. Response contains the prediction AND all features that led to it (8.4).

Run:
    python src/inference/hw_app.py
"""

import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# make src/ importable (hw_config, training_pipeline)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hw_config import (  # noqa: E402
    DATA_FILE,
    FEATURES_LOCAL_CSV,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MODEL_BUCKET,
    MODEL_FILE_NAME,
    MODEL_NAME,
    RT_FEATURES,
    login,
)

app = Flask(__name__)

_model = None
_fv = None
_codes = None


def _category_codes() -> dict:
    """Map non-numeric feature values to their training-time category codes.

    The training pipeline encodes non-numeric columns via
    .astype("category").cat.codes on the FULL dataset, so the codes follow
    the sorted unique values. Encoding a single-row vector would produce
    different codes (e.g. "Herbst" -> 0 instead of 1), so the mapping is
    computed from the exact dataset used for training (FEATURES_LOCAL_CSV).
    """
    global _codes
    if _codes is None:
        path = FEATURES_LOCAL_CSV if FEATURES_LOCAL_CSV.exists() else DATA_FILE
        df = pd.read_csv(path)
        _codes = {}
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                vals = sorted(df[col].astype(str).unique())
                _codes[col] = {v: i for i, v in enumerate(vals)}
    return _codes


def _encode_vector(vector: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `vector` with non-numeric columns integer-coded.

    Must mirror `encode_features` in src/hw_training_pipeline.py so the model
    receives the same representation it was trained on. The original (string)
    values are kept separately for the response.
    """
    X = vector.copy()
    for col, mapping in _category_codes().items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping)
    return X


def _get_fv():
    """Feature view from the Hopsworks Feature Store (lazy, once per process)."""
    global _fv
    if _fv is None:
        project = login()
        fs = project.get_feature_store()
        _fv = fs.get_feature_view(FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    return _fv


def _default_date() -> str:
    """Latest day that exists in the feature store.

    src/hw_feature_pipeline.py upserts the exact dataset it stores into the
    feature group (including the most recent dates, whose target label is
    NULL) and writes it to FEATURES_LOCAL_CSV, so its max date is the latest
    date available for /predict.
    """
    if FEATURES_LOCAL_CSV.exists():
        dates = pd.read_csv(FEATURES_LOCAL_CSV, usecols=["date"])["date"]
    else:
        dates = pd.read_csv(DATA_FILE, usecols=["date"])["date"]
    return pd.to_datetime(dates).max().strftime("%Y-%m-%d")


def _load_model():
    """Model from the Hopsworks Model Registry (latest version), S3 fallback."""
    global _model
    if _model is not None:
        return _model

    # 1) Hopsworks Model Registry (latest version of the model)
    try:
        project = login()
        registry = project.get_model_registry()
        models = registry.get_models(MODEL_NAME)
        if models:
            latest = max(models, key=lambda m: m.version)
            folder = latest.download()
            _model = joblib.load(os.path.join(folder, MODEL_FILE_NAME))
            print(f"Model '{MODEL_NAME}' v{latest.version} loaded from the Hopsworks Model Registry")
            return _model
        print(f"No model '{MODEL_NAME}' in the Model Registry, falling back to S3")
    except Exception as e:
        print(f"Hopsworks Model Registry not reachable ({e}), falling back to S3")

    # 2) S3 fallback (models from src/training_pipeline.py and the legacy app)
    import boto3
    from io import BytesIO

    s3 = boto3.client("s3")
    for key in (MODEL_FILE_NAME, "model.pkl"):
        try:
            response = s3.get_object(Bucket=MODEL_BUCKET, Key=key)
            _model = joblib.load(BytesIO(response["Body"].read()))
            print(f"Model loaded from s3://{MODEL_BUCKET}/{key}")
            return _model
        except Exception as e:
            print(f"s3://{MODEL_BUCKET}/{key} not available: {e}")
    raise RuntimeError("No model found in the Hopsworks Model Registry or on S3")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    # 1) live data / user input (real-time features, optional)
    passed_features = {
        k: float(v) for k, v in data.items() if k in RT_FEATURES and v is not None
    }

    date = str(data.get("date") or _default_date())
    try:
        entry_date = pd.to_datetime(date)
    except ValueError:
        return jsonify({"error": f"invalid date: {date!r} (expected e.g. 2026-10-02)"}), 400

    try:
        # 2) feature vector from the Hopsworks online feature store
        fv = _get_fv()
        vector = fv.get_feature_vector(
            entry={"date": entry_date},
            passed_features=passed_features or None,
            return_type="pandas",
        )
        row = vector.to_dict(orient="records")[0]
        missing = [k for k, v in row.items() if v is None]
        if missing:
            return (
                jsonify(
                    {
                        "error": (
                            f"no feature vector for date {entry_date.strftime('%Y-%m-%d')} "
                            f"in the feature store (missing: {', '.join(missing)}); "
                            f"latest available date: {_default_date()}"
                        )
                    }
                ),
                404,
            )
        # 3) prediction with the model from the Model Registry (or S3 fallback)
        #    (non-numeric features are integer-coded exactly as in training)
        model = _load_model()
        prediction = float(model.predict(_encode_vector(vector))[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # 4) prediction + all features that led to it
    result = {
        "date": entry_date.strftime("%Y-%m-%d"),
        "prediction_next_week_price": prediction,
        # keep non-numeric features (e.g. season) as strings
        "features": {
            k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()
        },
    }
    if passed_features:
        result["passed_features"] = passed_features
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
