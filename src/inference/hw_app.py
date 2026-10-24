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

GET /rhein
    Current Rhine gauge snapshot (Basel-Rheinhalle + Kaub bottleneck) written by
    src/collect_rhein_levels.py to s3://oilprice-features/raw/rhein_snapshot.json
    (local fallback data/rhein/rhein_snapshot.json). Gauges are the only Rhine
    data that is realtime; the cargo bulletins are lagged by 4-7 weeks.

Run:
    python src/inference/hw_app.py
"""

import json
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
    FORWARD_FEATURES,
    FORWARD_FEATURE_VIEW_NAME,
    FORWARD_FEATURE_VIEW_VERSION,
    FORWARD_HORIZONS,
    FORWARD_MODEL_FILE_NAME,
    FORWARD_MODEL_NAME,
    MODEL_BUCKET,
    MODEL_FILE_NAME,
    MODEL_NAME,
    RT_FEATURES,
    login,
)
from model_artifact import read_model_contract, read_model_features  # noqa: E402
from model_contract import (  # noqa: E402
    DELTA_TARGET,
    FORWARD_DELTA_TARGET,
    FORWARD_ORIGIN_PRICE,
    validate_features,
)

app = Flask(__name__)

_model = None
_fv = None
_codes = None
_forward_model = None
_forward_fv = None
_model_features = None
_model_contract = None
_forward_model_features = None
_forward_model_contract = None


def _category_codes() -> dict:
    """Map non-numeric feature values to their training-time category codes.

    LEGACY path, only for a forward artifact that was trained before the model
    contract existed (see `_prepare_forward_vector`). The current pipelines one-hot
    encode the raw category strings inside the persisted Pipeline, so integer
    coding a contract-aware artifact would map every category onto the wrong block.

    The codes follow the sorted unique values of the FULL training dataset, so the
    mapping is computed from that dataset (FEATURES_LOCAL_CSV) rather than from a
    single-row vector, which would produce different codes (e.g. "Herbst" -> 0
    instead of 1).
    """
    global _codes
    if _codes is None:
        _codes = {}

        for path in (FEATURES_LOCAL_CSV, FEATURES_LOCAL_CSV.with_name("forward_features.csv")):
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    vals = sorted(df[col].astype(str).unique())
                    _codes[col] = {v: i for i, v in enumerate(vals)}

        if DATA_FILE.exists() and "season" not in _codes:
            df = pd.read_csv(DATA_FILE)
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    vals = sorted(df[col].astype(str).unique())
                    _codes[col] = {v: i for i, v in enumerate(vals)}

        if "target_season" not in _codes:
            vals = sorted(["Frühling", "Herbst", "Sommer", "Winter"])
            _codes["target_season"] = {v: i for i, v in enumerate(vals)}

    return _codes


def _encode_vector(vector: pd.DataFrame) -> pd.DataFrame:
    """Integer-code non-numeric columns (forward model only).

    The weekly model no longer goes through here: its pipeline one-hot encodes
    `season` from the raw strings, so integer-coding it here would silently map
    every category onto the wrong one-hot block. Only the forward model, which is
    still trained with integer codes, uses this helper.
    """
    X = vector.copy()
    for col, mapping in _category_codes().items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping)
    return X


def _select_model_features(vector: pd.DataFrame, features: list | None) -> pd.DataFrame:
    """Select exactly the columns the registered model was trained on.

    The feature views gained the Rhine/market driver columns in v2, so the
    column set of a view is not a stable contract - the model artifact carries
    its own feature list (src/model_artifact.py) and that list wins here.
    """
    if not features:
        return vector
    missing = [name for name in features if name not in vector.columns]
    if missing:
        raise ValueError(f"feature vector lacks features the model was trained on: {missing}")
    return vector[list(features)]


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
    global _model, _model_features, _model_contract
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
            _model_features = read_model_features(folder)
            _model_contract = read_model_contract(folder)
            print(
                f"Model '{MODEL_NAME}' v{latest.version} loaded from the Hopsworks Model Registry"
                + (f" ({len(_model_features)} features)" if _model_features else "")
                + (f", target={_model_contract.target}" if _model_contract else "")
            )
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


def _latest_origin_date() -> pd.Timestamp:
    """Latest raw-price date available for forward forecasts.

    In production the online feature store is fresher than the CSVs baked into
    the inference image. Therefore, for future target dates we first try to
    discover the latest weekly feature-store date by probing backwards from the
    requested date. If that is not possible, we fall back to the local CSVs.
    """
    forward_path = FEATURES_LOCAL_CSV.with_name("forward_features.csv")
    if forward_path.exists():
        dates = pd.read_csv(forward_path, usecols=["origin_date"])["origin_date"]
        return pd.to_datetime(dates).max().normalize()

    if DATA_FILE.exists():
        dates = pd.read_csv(DATA_FILE, usecols=["date"])["date"]
    elif FEATURES_LOCAL_CSV.exists():
        dates = pd.read_csv(FEATURES_LOCAL_CSV, usecols=["date"])["date"]
    else:
        raise FileNotFoundError("No local data file available to determine origin date.")

    return pd.to_datetime(dates).max().normalize()


def _latest_weekly_date_before(entry_date: pd.Timestamp, max_lookback_days: int) -> pd.Timestamp | None:
    """Find the latest date <= entry_date that has a usable weekly feature vector.

    Hopsworks can return an all-NULL vector for dates that are not present in
    the online feature store, so `len(vector) > 0` alone is not enough.
    """
    try:
        fv = _get_fv()
    except Exception:
        return None

    for offset in range(1, max_lookback_days + 1):
        candidate = entry_date.normalize() - pd.Timedelta(days=offset)
        try:
            vector = fv.get_feature_vector(
                entry={"date": candidate},
                return_type="pandas",
            )
            if vector is None or len(vector) == 0:
                continue
            if vector.notna().any(axis=1).iloc[0]:
                return candidate
        except Exception:
            continue
    return None


def _get_forward_fv():
    """Forward feature view from the Hopsworks Feature Store."""
    global _forward_fv
    if _forward_fv is None:
        project = login()
        fs = project.get_feature_store()
        _forward_fv = fs.get_feature_view(
            FORWARD_FEATURE_VIEW_NAME, version=FORWARD_FEATURE_VIEW_VERSION
        )
    return _forward_fv


def _load_forward_model():
    """Forward model from the Hopsworks Model Registry, S3 fallback.

    The contract sidecar (src/model_artifact.py) decides what the artifact's
    number means: `forward_target_week_change` (a delta that has to be added to
    the origin price) or the price level itself (pre-contract artifacts).
    """
    global _forward_model, _forward_model_features, _forward_model_contract
    if _forward_model is not None:
        return _forward_model

    try:
        project = login()
        registry = project.get_model_registry()
        models = registry.get_models(FORWARD_MODEL_NAME)
        if models:
            latest = max(models, key=lambda m: m.version)
            folder = latest.download()
            _forward_model = joblib.load(os.path.join(folder, FORWARD_MODEL_FILE_NAME))
            _forward_model_features = read_model_features(folder)
            _forward_model_contract = read_model_contract(folder)
            target = (
                _forward_model_contract.target
                if _forward_model_contract is not None
                else "unknown (artifact without contract sidecar)"
            )
            print(
                f"Model '{FORWARD_MODEL_NAME}' v{latest.version} loaded "
                "from the Hopsworks Model Registry"
                + (f" ({len(_forward_model_features)} features)" if _forward_model_features else "")
                + f", target {target}"
            )
            return _forward_model
        print(f"No model '{FORWARD_MODEL_NAME}' in the Model Registry, falling back to S3")
    except Exception as e:
        print(f"Hopsworks Model Registry not reachable ({e}), falling back to S3")

    import boto3
    from io import BytesIO

    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=MODEL_BUCKET, Key=FORWARD_MODEL_FILE_NAME)
        _forward_model = joblib.load(BytesIO(response["Body"].read()))
        # An S3 artifact carries no contract sidecar, so it is served with the
        # legacy semantics: the pickle predicts the price level directly.
        _forward_model_contract = None
        print(f"Forward model loaded from s3://{MODEL_BUCKET}/{FORWARD_MODEL_FILE_NAME}")
        return _forward_model
    except Exception as e:
        print(f"s3://{MODEL_BUCKET}/{FORWARD_MODEL_FILE_NAME} not available: {e}")
    raise RuntimeError("No forward model found in the Hopsworks Model Registry or on S3")


def _prepare_forward_vector(vector: pd.DataFrame, contract):
    """Encode the forward feature vector the way this artifact was trained.

    A contract-aware forward artifact one-hot encodes `target_season` from the raw
    strings inside its persisted Pipeline, so the vector is validated against the
    contract and passed through unchanged. Only an artifact that predates the
    contract was trained on integer category codes; that path is kept so the model
    currently in the registry keeps serving until it is retrained.
    """
    if contract is None:
        return _encode_vector(vector)
    validate_features(vector, contract)
    return vector


def _forward_passed_features(passed_features: dict) -> dict:
    """Map user-provided live RT features to forward origin features."""
    out = {}
    if "current_price" in passed_features:
        out["origin_current_price"] = passed_features["current_price"]
    if "current_demand" in passed_features:
        out["origin_current_demand"] = passed_features["current_demand"]
    return out


def _forward_feature_vector(entry_date: pd.Timestamp, passed_features: dict):
    max_horizon = max(FORWARD_HORIZONS)
    fv = _get_forward_fv()
    forward_passed = _forward_passed_features(passed_features) or None

    def try_origin(origin_date: pd.Timestamp):
        horizon = int((entry_date.normalize() - origin_date).days)
        if horizon <= 0:
            return None, origin_date, horizon, "requested date is not after the latest known origin date"
        if horizon > max_horizon:
            return (
                None,
                origin_date,
                horizon,
                f"requested date is {horizon} days beyond the latest known origin date; "
                f"maximum supported forward horizon is {max_horizon} days",
            )
        try:
            vector = fv.get_feature_vector(
                entry={
                    "target_date": entry_date.normalize(),
                    "origin_date": origin_date,
                },
                passed_features=forward_passed,
                return_type="pandas",
            )
        except Exception as e:
            return None, origin_date, horizon, str(e)

        if vector is None or len(vector) == 0:
            return (
                None,
                origin_date,
                horizon,
                f"no forward feature vector for target_date "
                f"{entry_date.strftime('%Y-%m-%d')} and forecast_horizon_days {horizon}",
            )
        return vector, origin_date, horizon, None

    # Prefer the freshest origin date from the online weekly feature store.
    # This is the normal production case after the daily feature pipelines have
    # refreshed both the weekly and forward feature views.
    probed_origin = _latest_weekly_date_before(entry_date, max_horizon)
    if probed_origin is not None:
        vector, probed_origin, horizon, probe_error = try_origin(probed_origin)
        if vector is not None:
            return vector, probed_origin, horizon, None
        error = probe_error

    # If the online forward rows are stale or missing, fall back to the latest
    # origin date known from the local data files baked into the image.
    try:
        local_origin = _latest_origin_date()
        vector, origin_date, horizon, local_error = try_origin(local_origin)
        if vector is not None:
            return vector, origin_date, horizon, None
        error = local_error or error
    except Exception as e:
        error = str(e) or error
        if probed_origin is None:
            origin_date = entry_date.normalize()
            horizon = 0
        else:
            origin_date = probed_origin

    return None, origin_date, horizon, error


def _predict_forward(entry_date: pd.Timestamp, passed_features: dict):
    try:
        vector, origin_date, horizon, error = _forward_feature_vector(entry_date, passed_features)
    except Exception as e:
        return None, {
            "error": str(e),
            "requested_date": entry_date.strftime("%Y-%m-%d"),
        }

    if vector is None:
        return None, {
            "error": error,
            "requested_date": entry_date.strftime("%Y-%m-%d"),
            "origin_date": origin_date.strftime("%Y-%m-%d"),
            "forecast_horizon_days": horizon,
        }

    row = vector.to_dict(orient="records")[0]
    missing = [name for name in FORWARD_FEATURES if row.get(name) is None]
    if missing and len(missing) == len(FORWARD_FEATURES):
        return None, {
            "error": (
                f"no forward feature vector for target_date "
                f"{entry_date.strftime('%Y-%m-%d')} and forecast_horizon_days {horizon}"
            ),
            "requested_date": entry_date.strftime("%Y-%m-%d"),
            "origin_date": origin_date.strftime("%Y-%m-%d"),
            "forecast_horizon_days": horizon,
        }
    if missing:
        return None, {
            "error": (
                f"incomplete forward feature vector for target_date "
                f"{entry_date.strftime('%Y-%m-%d')} and forecast_horizon_days {horizon} "
                f"(missing: {', '.join(missing)})"
            ),
            "requested_date": entry_date.strftime("%Y-%m-%d"),
            "origin_date": origin_date.strftime("%Y-%m-%d"),
            "forecast_horizon_days": horizon,
        }

    model = _load_forward_model()
    model_features = _forward_model_features or FORWARD_FEATURES
    contract = _forward_model_contract
    X = _select_model_features(vector, model_features)
    raw = float(model.predict(_prepare_forward_vector(X, contract))[0])

    # What the raw number means is decided by the artifact, not by the app:
    # the current forward model predicts `forward_target_week_change`, the change
    # from the price known at origin_date, so the served price is reconstructed as
    # origin_current_price + delta. An artifact that predates the contract predicted
    # the level directly and is served unchanged.
    origin_price = float(row[FORWARD_ORIGIN_PRICE])
    if contract is not None and contract.target == FORWARD_DELTA_TARGET:
        prediction = origin_price + raw
        predicted_change = raw
        target_semantics = FORWARD_DELTA_TARGET
    else:
        prediction = raw
        predicted_change = prediction - origin_price
        target_semantics = (
            contract.target
            if contract is not None
            else "level (pre-contract artifact)"
        )

    # the response must list the columns the model was really trained on: since
    # v2 the Rhine/market drivers are part of the model, so reporting only
    # FORWARD_FEATURES would hide the driver values behind the prediction.
    reported = set(model_features)
    result = {
        "date": entry_date.strftime("%Y-%m-%d"),
        "requested_date": entry_date.strftime("%Y-%m-%d"),
        "origin_date": origin_date.strftime("%Y-%m-%d"),
        "forecast_horizon_days": horizon,
        "prediction_next_week_price": prediction,
        "predicted_week_change": predicted_change,
        "target": target_semantics,
        "model": "forward_forecast",
        "features": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in row.items()
            if k in reported
        },
    }
    if passed_features:
        result["passed_features"] = passed_features
    return result, None


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
        records = vector.to_dict(orient="records")
        if not records:
            forward_result, forward_error = _predict_forward(entry_date, passed_features)
            if forward_result is not None:
                return jsonify(forward_result)
            return (
                jsonify(
                    {
                        "error": (
                            f"no feature vector for date {entry_date.strftime('%Y-%m-%d')} "
                            f"in the feature store; latest available date: {_default_date()}"
                        ),
                        "forward_error": forward_error,
                    }
                ),
                404,
            )
        row = records[0]
        missing = [k for k, v in row.items() if v is None]
        if missing:
            forward_result, forward_error = _predict_forward(entry_date, passed_features)
            if forward_result is not None:
                return jsonify(forward_result)
            return (
                jsonify(
                    {
                        "error": (
                            f"no feature vector for date {entry_date.strftime('%Y-%m-%d')} "
                            f"in the feature store (missing: {', '.join(missing)}); "
                            f"latest available date: {_default_date()}"
                        ),
                        "forward_error": forward_error,
                    }
                ),
                404,
            )
        # 3) prediction with the model from the Model Registry (or S3 fallback).
        #    The weekly pipeline carries its own encoders: `season` stays a raw
        #    string and the point-in-time seasonal index is recomputed inside the
        #    model, so the vector is passed through unmodified - integer-coding it
        #    here would silently land every category on the wrong one-hot block.
        model = _load_model()
        contract = _model_contract
        features = _model_features or (contract.features if contract else None)
        X = _select_model_features(vector, features)
        if "date" not in X.columns:
            X = X.assign(date=entry_date)
        if contract is not None:
            validate_features(X, contract)

        raw = float(model.predict(X)[0])
        if contract is not None and contract.target == DELTA_TARGET:
            predicted_delta = raw
            prediction = float(row["current_price"]) + predicted_delta
        else:
            predicted_delta = None
            prediction = raw
    except Exception as e:
        forward_result, forward_error = _predict_forward(entry_date, passed_features)
        if forward_result is not None:
            return jsonify(forward_result)
        return jsonify({"error": str(e), "forward_error": forward_error}), 400

    # 4) prediction + all features that led to it
    result = {
        "date": entry_date.strftime("%Y-%m-%d"),
        "prediction_next_week_price": prediction,
        "model": "weekly",
        # keep non-numeric features (e.g. season) as strings
        "features": {
            k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()
        },
    }
    if predicted_delta is not None:
        result["predicted_week_change"] = predicted_delta
        result["prediction_rule"] = "current_price + predicted target_week_change"
    if passed_features:
        result["passed_features"] = passed_features
    return jsonify(result)


def _rhein_snapshot() -> dict:
    """Current Rhine gauge snapshot written by src/collect_rhein_levels.py.

    Read-only view of the realtime gauge situation (Basel-Rheinhalle and the
    Kaub bottleneck). Only the gauges are realtime data: the Port-of-Switzerland
    cargo figures are published 4-7 weeks after the reporting month and are
    therefore historical/lagged features, never an input to a live prediction.
    """
    bucket, key = "oilprice-features", "raw/rhein_snapshot.json"
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        try:
            import boto3

            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            return json.loads(body.decode("utf-8"))
        except Exception as e:
            print(f"s3://{bucket}/{key} not available: {e}")

    local = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "rhein", "rhein_snapshot.json")
    )
    with open(local, encoding="utf-8") as fh:
        return json.load(fh)


@app.route("/rhein", methods=["GET"])
def rhein():
    """Realtime Rhine gauge snapshot (Basel + Kaub bottleneck + low-water flag)."""
    try:
        return jsonify(_rhein_snapshot())
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
