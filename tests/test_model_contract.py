"""Contract sidecar + serving-path tests (no network, no Hopsworks)."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "inference"))

import hw_app  # noqa: E402
import training_pipeline as tp  # noqa: E402
from model_artifact import read_model_contract, read_model_features, write_model_features  # noqa: E402
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


def test_legacy_sidecar_without_contract_is_upgraded(tmp_path):
    write_model_features(tmp_path, "oil_price_model", ["current_price", "season"])
    assert read_model_features(tmp_path) == ["current_price", "season"]
    legacy = read_model_contract(tmp_path)
    assert legacy is not None
    assert legacy.target == LEVEL_TARGET  # pre-delta artifacts stay readable


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
