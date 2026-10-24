"""Chronological split, embargo and contract tests (no network, no S3)."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import training_pipeline as tp  # noqa: E402
from model_contract import (  # noqa: E402
    DELTA_TARGET,
    LEVEL_TARGET,
    TARGET_HORIZON_DAYS,
    ContractError,
    ModelContract,
    chronological_split,
    validate_features,
)


def _frame(n_days: int = 4000, start: str = "2010-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days, dtype="float64")
    price = 40.0 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 365.0)
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


# ---------------------------------------------------------------- split ----
def test_train_dates_strictly_before_test_dates():
    train, test, audit = chronological_split(_frame(), test_size=0.2)
    assert train["date"].max() < test["date"].min()


def test_train_label_window_does_not_overlap_test_feature_window():
    """The whole point of the embargo."""
    train, test, audit = chronological_split(_frame(), test_size=0.2)
    last_label = train["date"].max() + pd.Timedelta(days=TARGET_HORIZON_DAYS)
    assert last_label < test["date"].min()
    assert pd.Timestamp(audit["train_label_end_date"]) < pd.Timestamp(audit["test_start_date"])


def test_embargo_drops_exactly_horizon_rows():
    frame = _frame()
    n = len(frame)
    cut = int(n * 0.8)
    _, _, audit = chronological_split(frame, test_size=0.2)
    assert audit["dropped_by_embargo"] == TARGET_HORIZON_DAYS
    assert audit["train_rows"] == cut
    assert audit["test_rows"] == n - cut - TARGET_HORIZON_DAYS


def test_gap_between_train_and_test_is_at_least_the_horizon():
    train, test, audit = chronological_split(_frame(), test_size=0.2)
    gap = (test["date"].min() - train["date"].max()).days
    # Daily frame: cut + embargo leaves horizon + 1 clear days between the last
    # training feature row and the first test feature row.
    assert gap >= TARGET_HORIZON_DAYS
    assert (test["date"].min() - train["date"].max()).days == TARGET_HORIZON_DAYS + 1


def test_split_is_deterministic_and_sorted():
    frame = _frame().sample(frac=1.0, random_state=7)  # unsorted input
    train, test, _ = chronological_split(frame, test_size=0.2)
    assert train["date"].is_monotonic_increasing
    assert test["date"].is_monotonic_increasing
    assert train["date"].min() == frame["date"].min()


def test_missing_date_column_is_rejected():
    frame = _frame().drop(columns=["date"])
    with pytest.raises(ContractError, match="date"):
        chronological_split(frame, test_size=0.2)


def test_duplicate_dates_are_rejected():
    frame = _frame()
    frame.loc[10, "date"] = frame.loc[9, "date"]
    with pytest.raises(ContractError, match="duplicate"):
        chronological_split(frame, test_size=0.2)


def test_tiny_test_side_after_embargo_is_rejected():
    frame = _frame(n_days=1200)  # 240 test rows minus embargo is still > 100
    with pytest.raises(ContractError, match="too small"):
        chronological_split(frame, test_size=0.2, min_test_rows=240)


def test_prepare_data_returns_delta_target_and_keeps_season_as_string():
    frame = tp.prepare_data(_frame().drop(columns=[DELTA_TARGET]))
    assert DELTA_TARGET in frame.columns
    assert np.allclose(frame[DELTA_TARGET], frame[LEVEL_TARGET] - frame["current_price"])
    assert not pd.api.types.is_numeric_dtype(frame["season"])


def test_prepare_data_rejects_integer_coded_season():
    frame = _frame()
    frame["season"] = frame["season"].astype("category").cat.codes
    with pytest.raises(ContractError, match="season"):
        tp.prepare_data(frame)



# ------------------------------------------------------------ pipeline ----
def test_build_model_rejects_integer_coded_categoricals():
    frame = _frame()
    X = frame[tp.MODEL_INPUTS].copy()
    X["season"] = X["season"].astype("category").cat.codes
    with pytest.raises(ContractError, match="raw category strings"):
        tp.build_model(X)


def test_build_model_rejects_missing_categorical():
    frame = _frame().drop(columns=["season"])
    with pytest.raises(ContractError, match="Categorical features missing"):
        tp.build_model(frame[["date", "current_price", "price_mean_7d"]])


def test_pipeline_emits_no_duplicate_feature_names():
    """Guards the `base_numeric + ['seasonal_index']` duplication bug."""
    frame = _frame()
    model = tp.build_model(frame[tp.MODEL_INPUTS])
    model.fit(frame[tp.MODEL_INPUTS], frame[DELTA_TARGET])
    names = list(model.named_steps["preprocessing"].get_feature_names_out())
    assert len(names) == len(set(names)), names
    assert names.count("numeric__seasonal_index") == 1


def test_pipeline_overwrites_the_leaky_parquet_seasonal_index():
    frame = _frame()
    frame["seasonal_index"] = 0.111  # stale value shipped in the parquet
    model = tp.build_model(frame[tp.MODEL_INPUTS])
    model.fit(frame[tp.MODEL_INPUTS], frame[DELTA_TARGET])
    engineered = model.named_steps["feature_engineering"].transform(frame[tp.MODEL_INPUTS])
    assert not np.allclose(engineered["seasonal_index"], 0.111)
    assert engineered["seasonal_index"].std() > 0.01


def test_unseen_season_value_does_not_raise_and_encodes_all_zero():
    frame = _frame()
    model = tp.build_model(frame[tp.MODEL_INPUTS])
    model.fit(frame[tp.MODEL_INPUTS], frame[DELTA_TARGET])

    row = frame[tp.MODEL_INPUTS].iloc[[0]].copy()
    row["season"] = "Fruehling"  # ASCII normalisation from a serving client
    encoded = model.named_steps["preprocessing"].transform(row)
    assert encoded[:, :4].sum() == 0.0  # silently all-zero -> why the contract checks it
    assert len(model.predict(row)) == 1


def test_missing_feature_raises_from_the_pipeline():
    frame = _frame()
    model = tp.build_model(frame[tp.MODEL_INPUTS])
    model.fit(frame[tp.MODEL_INPUTS], frame[DELTA_TARGET])
    with pytest.raises(Exception):
        model.predict(frame[tp.MODEL_INPUTS].drop(columns=["price_mean_7d"]))


# ------------------------------------------------------------ contract ----
def test_contract_roundtrip_preserves_unicode_vocabulary():
    contract = ModelContract(features=tp.MODEL_INPUTS)
    restored = ModelContract.from_dict(contract.to_dict())
    assert restored.season_vocabulary == ("Winter", "Frühling", "Sommer", "Herbst")
    assert restored.target == DELTA_TARGET
    assert restored.horizon_days == TARGET_HORIZON_DAYS


def test_validate_features_rejects_missing_columns():
    frame = _frame()
    with pytest.raises(ContractError, match="missing model features"):
        validate_features(frame[tp.MODEL_INPUTS].drop(columns=["current_demand"]),
                          ModelContract(features=tp.MODEL_INPUTS))


def test_validate_features_rejects_integer_coded_season():
    frame = _frame()
    frame["season"] = frame["season"].astype("category").cat.codes
    with pytest.raises(ContractError, match="integer-coded"):
        validate_features(frame[tp.MODEL_INPUTS], ModelContract(features=tp.MODEL_INPUTS))


def test_validate_features_rejects_ascii_season():
    frame = _frame()
    frame.loc[0, "season"] = "Fruehling"
    with pytest.raises(ContractError, match="unseen season values"):
        validate_features(frame[tp.MODEL_INPUTS], ModelContract(features=tp.MODEL_INPUTS))


def test_serving_frame_matches_the_trained_contract():
    """The frame hw_app builds must be accepted by the trained pipeline."""
    frame = _frame()
    model = tp.build_model(frame[tp.MODEL_INPUTS])
    model.fit(frame[tp.MODEL_INPUTS], frame[DELTA_TARGET])

    contract = ModelContract(features=tp.MODEL_INPUTS)
    serving = frame[tp.MODEL_INPUTS].tail(5).copy()
    validate_features(serving, contract)

    level = frame["current_price"].to_numpy(dtype="float64")[-5:] + model.predict(serving)
    assert level.shape == (5,)
    assert np.isfinite(level).all()
