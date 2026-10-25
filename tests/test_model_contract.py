"""Contract sidecar + serving-path tests (no network, no Hopsworks)."""

import io
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "inference"))

import hw_app  # noqa: E402
import training_pipeline as tp  # noqa: E402
from model_artifact import (  # noqa: E402
    CONTRACT_VERSION,
    contract_from_payload,
    read_model_contract,
    read_model_features,
    sidecar_keys,
    write_model_features,
)
from model_contract import DELTA_TARGET, LEVEL_TARGET, ModelContract  # noqa: E402


def _frame(n_days: int = 3000) -> pd.DataFrame:
    dates = pd.date_range("2012-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype="float64")
    price = 60.0 + 0.01 * t + 2.0 * np.sin(2 * np.pi * t / 365.0)
    frame = pd.DataFrame({"date": dates, "current_price": price})
    frame["price_mean_7d"] = frame["current_price"].rolling(7).mean()
    frame["price_std_30d"] = frame["current_price"].rolling(30).std()
    frame["demand_mean_7d"] = 10.0
    frame["demand_mean_30d"] = 10.0
    frame["month"] = dates.month
    frame["season"] = pd.Series(dates.month).map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Frühling", 4: "Frühling",
         5: "Frühling", 6: "Sommer", 7: "Sommer", 8: "Sommer",
         9: "Herbst", 10: "Herbst", 11: "Herbst"}
    )
    frame["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    frame["seasonal_index"] = 1.0
    frame["price_mean_dow_4w"] = frame["current_price"].shift(7)
    frame["current_demand"] = 10.0
    frame["target_next_week_price"] = frame["current_price"].shift(7)
    frame[DELTA_TARGET] = frame[LEVEL_TARGET] - frame["current_price"]
    return frame.dropna().reset_index(drop=True)


@pytest.fixture(scope="module")
def trained():
    frame = _frame()
    model = tp.build_model(frame[tp.MODEL_INPUTS])
    model.fit(frame[tp.MODEL_INPUTS], frame[DELTA_TARGET])
    contract = ModelContract(features=tp.MODEL_INPUTS)
    return frame, model, contract


# ------------------------------------------------------------- sidecar ----
def test_sidecar_roundtrip_keeps_unicode_vocabulary(tmp_path, trained):
    _, _, contract = trained
    write_model_features(tmp_path, "oil_price_model", contract.features, contract)
    restored = read_model_contract(tmp_path)
    assert restored is not None
    assert "Frühling" in restored.season_vocabulary
    assert restored.target == DELTA_TARGET
    assert restored.horizon_days == 7
    assert restored.features == contract.features


def test_legacy_sidecar_without_contract_is_not_invented(tmp_path):
    """A pre-contract sidecar must NOT be upgraded into a contract.

    It lists only `features`, so nothing about it says how the artifact was
    encoded. Pretending otherwise makes the serving app pass raw category strings
    to an estimator that was fitted on integer codes, which fails with
    "could not convert string to float". Returning None routes the artifact onto
    the legacy path: integer-coded vector, prediction read as a price level.
    """
    write_model_features(tmp_path, "oil_price_model", ["current_price", "season"])
    assert read_model_features(tmp_path) == ["current_price", "season"]
    assert read_model_contract(tmp_path) is None


def test_contract_sidecar_is_marked_as_such(tmp_path, trained):
    """The marker that separates the two serving paths is written explicitly."""
    _, _, contract = trained
    write_model_features(tmp_path, "oil_price_model", contract.features, contract)
    payload = json.loads((tmp_path / "model_features.json").read_text(encoding="utf-8"))
    assert payload["contract_version"] == CONTRACT_VERSION
    assert read_model_contract(tmp_path) is not None


def test_pre_marker_contract_sidecar_is_still_a_contract(tmp_path, trained):
    """The forward artifact in the registry predates the `contract_version` marker.

    The writer that produced it emitted the whole contract dict - target,
    categorical_features, season_vocabulary - whenever the pipeline supplied a
    contract. Refusing such a sidecar would integer-code `target_season` and hand
    integers to a vocabulary-keyed OneHotEncoder, so the declared target is what
    separates it from a features-only legacy sidecar.
    """
    _, _, contract = trained
    write_model_features(tmp_path, "oil_price_forward_model", contract.features, contract)
    path = tmp_path / "model_features.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored.pop("contract_version")
    path.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")

    restored = read_model_contract(tmp_path)
    assert restored is not None
    assert restored.target == contract.target
    assert restored.features == contract.features
    assert "Frühling" in restored.season_vocabulary


def test_missing_sidecar_returns_none(tmp_path):
    assert read_model_features(tmp_path) is None
    assert read_model_contract(tmp_path) is None


# ------------------------------------------------------------- serving ----
def test_predict_endpoint_reconstructs_the_level(monkeypatch, trained):
    """hw_app must add current_price back onto the predicted delta."""
    frame, model, contract = trained
    row = frame.iloc[[-1]]

    class FakeFv:
        def get_feature_vector(self, entry, passed_features=None, return_type="pandas"):
            return row[tp.MODEL_INPUTS].copy()

    monkeypatch.setattr(hw_app, "_get_fv", lambda: FakeFv())
    monkeypatch.setattr(hw_app, "_load_model", lambda: model)
    monkeypatch.setattr(hw_app, "_model_features", contract.features)
    monkeypatch.setattr(hw_app, "_model_contract", contract)
    monkeypatch.setattr(hw_app, "_predict_forward", lambda *a, **k: (None, "stub"))

    client = hw_app.app.test_client()
    response = client.post(
        "/predict", json={"date": row["date"].iloc[0].strftime("%Y-%m-%d")}
    )
    assert response.status_code == 200
    body = response.get_json()
    delta = float(model.predict(row[tp.MODEL_INPUTS])[0])
    assert body["predicted_week_change"] == pytest.approx(delta)
    assert body["prediction_next_week_price"] == pytest.approx(
        float(row["current_price"].iloc[0]) + delta
    )
    assert body["prediction_rule"] == "current_price + predicted target_week_change"
    # season must reach the response as a string, never as a code
    assert body["features"]["season"] in {"Winter", "Frühling", "Sommer", "Herbst"}


def test_contract_from_payload_refuses_a_legacy_payload():
    """The same rule the registry path uses, applied to a raw bucket payload."""
    assert contract_from_payload({"model_name": "m", "features": ["current_price"]}) is None
    assert contract_from_payload(None) is None

    marked = ModelContract(features=["current_price", "season"]).to_dict()
    marked["contract_version"] = CONTRACT_VERSION
    restored = contract_from_payload(marked)
    assert restored is not None
    assert restored.features == ["current_price", "season"]


def test_sidecar_keys_cover_both_bucket_spellings():
    """The weekly export and the forward export spell the sidecar differently."""
    assert sidecar_keys("oil_price_model.pkl") == [
        "oil_price_model_features.json",
        "model_features.json",
    ]
    assert sidecar_keys("oil_price_forward_model.pkl") == [
        "oil_price_forward_model_features.json",
        "model_features.json",
    ]


class _FakeS3:
    """Minimal stand-in for the boto3 client used by the S3 fallback."""

    def __init__(self, objects: dict):
        self.objects = objects

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key].encode("utf-8"))}


def test_read_s3_sidecar_reads_the_contract_of_an_exported_model():
    payload = ModelContract(features=["current_price", "season"]).to_dict()
    payload["contract_version"] = CONTRACT_VERSION
    s3 = _FakeS3({"oil_price_model_features.json": json.dumps(payload)})
    contract, features = hw_app._read_s3_sidecar(s3, "oil_price_model.pkl")
    assert contract is not None
    assert features == ["current_price", "season"]


def test_read_s3_sidecar_keeps_features_but_refuses_legacy_contract():
    """A legacy export still says which columns it wants - only the contract is refused."""
    payload = {"model_name": "oil_price_model", "features": ["current_price", "season"]}
    s3 = _FakeS3({"model_features.json": json.dumps(payload)})
    contract, features = hw_app._read_s3_sidecar(s3, "oil_price_model.pkl")
    assert contract is None
    assert features == ["current_price", "season"]


def test_read_s3_sidecar_without_any_sidecar():
    contract, features = hw_app._read_s3_sidecar(_FakeS3({}), "oil_price_model.pkl")
    assert contract is None
    assert features is None


# ------------------------------------------------- legacy weekly artifact ----
LEGACY_WEEKLY_FEATURES = ["current_price", "season", "month", "price_mean_7d"]


def _legacy_weekly_model(frame: pd.DataFrame):
    """A model built the way the pre-contract pipeline built it.

    `encode_features()` integer-coded every non-numeric column before fitting, so
    the persisted ColumnTransformer has an empty categorical branch and `season`
    sits in the numeric passthrough - exactly the artifact that is live in the
    Model Registry today.
    """
    X = frame[LEGACY_WEEKLY_FEATURES].copy()
    X["season"] = X["season"].astype("category").cat.codes
    model = Pipeline(
        [
            (
                "columntransformer",
                ColumnTransformer([("numeric", "passthrough", LEGACY_WEEKLY_FEATURES)]),
            ),
            ("randomforestregressor", RandomForestRegressor(n_estimators=20, random_state=42)),
        ]
    )
    model.fit(X, frame[LEVEL_TARGET])
    return model


def test_legacy_weekly_artifact_is_integer_coded_and_read_as_a_level(monkeypatch):
    """Without a contract, the weekly vector must be encoded the legacy way.

    Feeding the raw strings to such an artifact is the failure this guards:
    "could not convert string to float: 'Herbst'".
    """
    frame = _frame()
    model = _legacy_weekly_model(frame)
    row = frame.iloc[[-1]]

    with pytest.raises(ValueError, match="could not convert string to float"):
        model.predict(row[LEGACY_WEEKLY_FEATURES])

    class FakeFv:
        def get_feature_vector(self, entry, passed_features=None, return_type="pandas"):
            return row[LEGACY_WEEKLY_FEATURES].copy()

    monkeypatch.setattr(hw_app, "_get_fv", lambda: FakeFv())
    monkeypatch.setattr(hw_app, "_load_model", lambda: model)
    monkeypatch.setattr(hw_app, "_model_features", LEGACY_WEEKLY_FEATURES)
    monkeypatch.setattr(hw_app, "_model_contract", None)
    monkeypatch.setattr(hw_app, "_predict_forward", lambda *a, **k: (None, "stub"))

    client = hw_app.app.test_client()
    response = client.post(
        "/predict", json={"date": row["date"].iloc[0].strftime("%Y-%m-%d")}
    )
    assert response.status_code == 200
    body = response.get_json()

    coded = row[LEGACY_WEEKLY_FEATURES].copy()
    coded["season"] = coded["season"].astype(str).map(hw_app._category_codes())
    assert body["prediction_next_week_price"] == pytest.approx(
        float(model.predict(coded)[0])
    )
    # A pre-contract artifact predicts the price level itself: no reconstruction.
    assert "predicted_week_change" not in body
    assert "prediction_rule" not in body
    # the response still reports the season as a string, not as a code
    assert body["features"]["season"] in {"Winter", "Frühling", "Sommer", "Herbst"}


def test_predict_endpoint_rejects_ascii_season(monkeypatch, trained):
    frame, model, contract = trained
    row = frame.iloc[[-1]].copy()
    row["season"] = "Fruehling"

    class FakeFv:
        def get_feature_vector(self, entry, passed_features=None, return_type="pandas"):
            return row[tp.MODEL_INPUTS].copy()

    monkeypatch.setattr(hw_app, "_get_fv", lambda: FakeFv())
    monkeypatch.setattr(hw_app, "_load_model", lambda: model)
    monkeypatch.setattr(hw_app, "_model_features", contract.features)
    monkeypatch.setattr(hw_app, "_model_contract", contract)
    monkeypatch.setattr(hw_app, "_predict_forward", lambda *a, **k: (None, "stub"))

    client = hw_app.app.test_client()
    response = client.post(
        "/predict", json={"date": row["date"].iloc[0].strftime("%Y-%m-%d")}
    )
    assert response.status_code == 400
    assert "unseen season values" in response.get_json()["error"]
