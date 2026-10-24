"""Forward-contract tests: split, baselines, delta target, serving semantics.

No network and no Hopsworks: the forward training module is exercised on a
synthetic frame that has the same column layout as the feature view.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "inference"))

import hw_app  # noqa: E402
import hw_config  # noqa: E402
import hw_forward_training_pipeline as fwd  # noqa: E402
from model_contract import (  # noqa: E402
    FORWARD_CATEGORICAL_FEATURES,
    FORWARD_DELTA_TARGET,
    FORWARD_LEVEL_TARGET,
    FORWARD_MAX_HORIZON_DAYS,
    FORWARD_ORIGIN_PRICE,
    ContractError,
    ModelContract,
    forward_chronological_split,
    forward_contract,
    forward_persistence_baselines,
    validate_features,
)

# A reduced but faithful feature set: the full list depends on which driver
# feature groups exist in a given project, the contract rules do not.
FEATURES = [
    "origin_current_price",
    "origin_price_mean_7d",
    "target_season",
    "target_month_sin",
    "forecast_horizon_days",
]
HORIZONS = (1, 7, 14, 30, 60, 90)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Frühling"
    if month in (6, 7, 8):
        return "Sommer"
    return "Herbst"


def _forward_frame(n_origins: int = 420) -> pd.DataFrame:
    """Synthetic forward rows with a real, horizon-dependent signal.

    The label is the level; the model has to learn the change. The signal is only
    observable from the horizon and the target season, so a model that ignores
    them cannot beat persistence.
    """
    rng = np.random.default_rng(7)
    origins = pd.date_range("2019-01-01", periods=n_origins, freq="D")
    rows = []
    for i, origin in enumerate(origins):
        price = 80.0 + 0.05 * i + 3.0 * np.sin(2 * np.pi * i / 365.0)
        for horizon in HORIZONS:
            target_date = origin + pd.Timedelta(days=horizon)
            signal = (
                0.02 * horizon
                + 1.5 * np.sin(2 * np.pi * target_date.month / 12)
                + rng.normal(0.0, 0.4)
            )
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": target_date,
                    "forecast_horizon_days": horizon,
                    "origin_current_price": price,
                    "origin_price_mean_7d": price - 0.1,
                    "target_season": _season(target_date.month),
                    "target_month_sin": float(np.sin(2 * np.pi * target_date.month / 12)),
                    FORWARD_LEVEL_TARGET: price + signal,
                }
            )
    return fwd.add_forward_delta(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _forward_frame()


# ------------------------------------------------------------- contract ----
def test_feature_view_label_and_contract_level_target_agree():
    """FORWARD_TARGET is a feature-group column name, not a free variable."""
    assert hw_config.FORWARD_TARGET == FORWARD_LEVEL_TARGET
    assert hw_config.FORWARD_DELTA_TARGET == FORWARD_DELTA_TARGET
    assert FORWARD_DELTA_TARGET != FORWARD_LEVEL_TARGET


def test_forward_contract_anchors_on_origin_date():
    contract = forward_contract(FEATURES)
    assert contract.target == FORWARD_DELTA_TARGET
    assert contract.level_target == FORWARD_LEVEL_TARGET
    assert contract.date_column == "origin_date"
    assert contract.categorical_features == FORWARD_CATEGORICAL_FEATURES
    assert contract.features == FEATURES


def test_delta_target_is_the_change_relative_to_the_origin_price(frame):
    expected = frame[FORWARD_LEVEL_TARGET] - frame[FORWARD_ORIGIN_PRICE]
    assert frame[FORWARD_DELTA_TARGET].equals(expected)


# ---------------------------------------------------------------- split ----
def test_split_never_shares_an_origin_between_train_and_test(frame):
    train, test, audit = forward_chronological_split(frame, test_size=0.2)
    train_origins = set(pd.to_datetime(train["origin_date"]))
    test_origins = set(pd.to_datetime(test["origin_date"]))
    assert not train_origins & test_origins
    assert max(train_origins) < min(test_origins)
    assert audit["train_rows"] + audit["test_rows"] == len(frame) - audit["dropped_by_embargo"]


def test_split_embargoes_the_long_horizon_label_windows(frame):
    """A 90-day training row labels a week 97 days past its origin; it must go."""
    train, _test, audit = forward_chronological_split(frame, test_size=0.2)
    label_end = (
        pd.to_datetime(train["origin_date"])
        + pd.to_timedelta(train["forecast_horizon_days"], unit="D")
        + pd.Timedelta(days=7)
    )
    assert label_end.max() < pd.Timestamp(audit["test_start_date"])
    assert audit["embargo_days"] == 7 + FORWARD_MAX_HORIZON_DAYS
    assert audit["dropped_by_embargo"] > 0


def test_split_is_stable_when_the_frame_arrives_shuffled(frame):
    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    a = forward_chronological_split(frame, test_size=0.2)[2]
    b = forward_chronological_split(shuffled, test_size=0.2)[2]
    assert a["train_rows"] == b["train_rows"]
    assert a["test_start_date"] == b["test_start_date"]


def test_split_requires_the_forward_keys(frame):
    with pytest.raises(ContractError, match="origin_date"):
        forward_chronological_split(frame.drop(columns=["origin_date"]), test_size=0.2)
    with pytest.raises(ContractError, match="forecast_horizon_days"):
        forward_chronological_split(frame.drop(columns=["forecast_horizon_days"]), test_size=0.2)


def test_split_refuses_a_frame_that_is_too_small(frame):
    with pytest.raises(ContractError):
        forward_chronological_split(frame.head(50), test_size=0.2)


# ------------------------------------------------------------ baselines ----
def test_forward_baselines_use_the_origin_price(frame):
    test = frame.tail(500)
    baselines = forward_persistence_baselines(test)
    truth = test[FORWARD_LEVEL_TARGET].to_numpy()
    assert baselines[FORWARD_ORIGIN_PRICE] == pytest.approx(
        float(np.mean((truth - test[FORWARD_ORIGIN_PRICE].to_numpy()) ** 2))
    )
    # "no signal at all, but the average drift is known" must also be reported
    assert "constant_delta" in baselines


# ------------------------------------------------------------ training ----
@pytest.fixture(scope="module")
def fitted(frame):
    train, test, _ = forward_chronological_split(frame, test_size=0.2)
    model = fwd.build_forward_model(FEATURES, FORWARD_CATEGORICAL_FEATURES)
    model.fit(train[FEATURES], train[FORWARD_DELTA_TARGET])
    return model, train, test


def test_model_is_fitted_on_the_delta_not_the_level(fitted):
    model, train, _ = fitted
    predicted = model.predict(train[FEATURES])
    # a level model would predict ~80-100 here; a delta model predicts a small drift
    assert abs(float(np.mean(predicted))) < 10.0
    assert float(np.max(predicted)) < 20.0


def test_delta_model_beats_persistence_on_the_reconstructed_level(fitted):
    model, train, test = fitted
    metrics = fwd.evaluate_forward_model(model, train, test, input_columns=FEATURES)
    assert metrics["skill_vs_persistence"] < 1.0
    assert metrics["level_mse"] < metrics["baselines"][FORWARD_ORIGIN_PRICE]
    # every horizon bucket must be reported, including the far end
    assert set(metrics["by_horizon"]) == {"1_7d", "8_30d", "31_60d", "61_90d"}
    assert metrics["by_horizon"]["61_90d"]["skill_vs_persistence"] < 1.0


def test_level_metrics_reconstruct_origin_plus_delta(fitted):
    model, train, test = fitted
    metrics = fwd.evaluate_forward_model(model, train, test, input_columns=FEATURES)
    delta = model.predict(test[FEATURES])
    level = test[FORWARD_ORIGIN_PRICE].to_numpy() + delta
    truth = test[FORWARD_LEVEL_TARGET].to_numpy()
    assert metrics["level_mse"] == pytest.approx(float(np.mean((truth - level) ** 2)))


def test_build_forward_model_rejects_a_missing_categorical():
    with pytest.raises(ContractError, match="categorical"):
        fwd.build_forward_model(["origin_current_price"], FORWARD_CATEGORICAL_FEATURES)


# ------------------------------------------------------------- serving ----
def test_validate_features_checks_target_season_not_only_season(frame):
    contract = forward_contract(FEATURES)
    validate_features(frame[FEATURES], contract)  # raw strings: fine

    coded = frame[FEATURES].copy()
    coded["target_season"] = coded["target_season"].astype("category").cat.codes
    with pytest.raises(ContractError, match="integer-coded"):
        validate_features(coded, contract)

    wrong = frame[FEATURES].copy()
    wrong["target_season"] = "Fruehling"
    with pytest.raises(ContractError, match="unseen target_season values"):
        validate_features(wrong, contract)


def test_validate_features_uses_the_contract_date_column(frame):
    """The anchor check follows the contract instead of a hard-coded 'date'."""
    strict = ModelContract(features=FEATURES, date_column="origin_date")
    with pytest.raises(ContractError, match="origin_date"):
        validate_features(frame[FEATURES], strict)

    # The forward artifact is served without the anchor: origin_date is a lookup
    # key and a split anchor, never a model feature.
    contract = forward_contract(FEATURES)
    assert contract.date_column == "origin_date"
    assert contract.requires_date is False
    validate_features(frame[FEATURES], contract)


def test_prepare_forward_vector_keeps_strings_for_a_contract_artifact(frame):
    contract = forward_contract(FEATURES)
    out = hw_app._prepare_forward_vector(frame[FEATURES].head(3), contract)
    assert out["target_season"].iloc[0] in {"Winter", "Frühling", "Sommer", "Herbst"}


def test_prepare_forward_vector_integer_codes_a_legacy_artifact(frame):
    out = hw_app._prepare_forward_vector(frame[FEATURES].head(3), None)
    assert pd.api.types.is_numeric_dtype(out["target_season"])


def test_sidecar_roundtrip_keeps_the_forward_anchor(tmp_path, frame):
    """A registered forward artifact must come back anchored on `origin_date`.

    The serving app rebuilds the contract from the sidecar and validates the
    online feature vector against it. A contract that round-trips to the weekly
    anchor rejects every forward request with "frame is missing 'date'", which is
    exactly how a correctly trained delta model 500s in production.
    """
    from model_artifact import read_model_contract, write_model_features

    contract = forward_contract(FEATURES)
    model_dir = tmp_path / hw_config.FORWARD_MODEL_NAME
    model_dir.mkdir()
    write_model_features(model_dir, hw_config.FORWARD_MODEL_NAME, FEATURES, contract)

    restored = read_model_contract(model_dir)
    assert restored is not None
    assert restored.target == FORWARD_DELTA_TARGET
    assert restored.requires_date is False
    assert restored.date_column == "origin_date"

    # The check hw_app performs on a live vector has to pass with the restored
    # contract, not just with the one built in-process.
    validate_features(frame[FEATURES], restored)
