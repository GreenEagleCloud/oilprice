"""Single source of truth for the model input/output contract.

Shared by the training pipelines (src/training_pipeline.py,
src/hw_training_pipeline.py) and the inference app (src/inference/hw_app.py).

Why this module exists
----------------------
A joblib pickle carries the fitted encoders but nothing about *how* the frame was
built: which columns are required, which are categorical, what the target means,
and how far ahead it looks. Those facts lived in three places (notebook 04,
hw_config.py, hw_app.py) and drifted. This module pins them, and
`chronological_split` puts the embargo rule in exactly one place so no caller can
reintroduce the overlap between a training label window and a test feature window.

Forward contract
----------------
The forward model forecasts the week-after price level at horizons of 1..90 days
measured from an *origin* date, so its rows are keyed by
(target_date, origin_date, forecast_horizon_days) instead of a single `date`.
It is trained on the same kind of stationary quantity as the weekly model: the
change relative to the price known at the origin date, not the absolute level.
RandomForest cannot extrapolate a level, and a level-trained forward model loses
to the baseline "the price at the origin date is still the price 90 days later".
`forward_chronological_split` and `forward_persistence_baselines` apply the same
leak and gate rules to that row layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Contract constants
# --------------------------------------------------------------------------
TARGET_HORIZON_DAYS = 7          # the label looks 7 days ahead of the feature row
CATEGORICAL_FEATURES = ("season",)
LEVEL_TARGET = "target_next_week_price"
DELTA_TARGET = "target_week_change"   # LEVEL_TARGET - current_price (stationary)

# Promotion gates
MIN_TRAIN_ROWS = 1000
MIN_TEST_ROWS = 100
MAX_SKILL_VS_PERSISTENCE = 1.0   # model MSE must be strictly better than persistence

SEASON_VOCABULARY = ("Winter", "Frühling", "Sommer", "Herbst")

# --------------------------------------------------------------------------
# Forward (multi-horizon) contract
# --------------------------------------------------------------------------
# Level label stored in the forward feature view: the mean price of the week
# *after* target_date, i.e. mean(target_date+1 .. target_date+7). The name is part
# of the Hopsworks feature-group schema and must not change.
FORWARD_LEVEL_TARGET = "target_price_week_after_target_date"
# What the forward model regresses on: the change from the price known at
# origin_date to that week-after level. Stationary, so the forest can extrapolate;
# the level is rebuilt at serving time as
# origin_current_price + predicted forward_target_week_change.
FORWARD_DELTA_TARGET = "forward_target_week_change"
# Persistence reference for a forward forecast: the last price known when the
# forecast is made. The forward frame has no `current_price` column - there it is
# `origin_current_price`.
FORWARD_ORIGIN_PRICE = "origin_current_price"
FORWARD_CATEGORICAL_FEATURES = ("target_season",)
# Longest forecast horizon the forward feature view materialises
# (hw_config.FORWARD_HORIZONS). The split embargo needs it: a training row with a
# 90-day horizon carries a label 90 + 7 days past its origin date.
FORWARD_MAX_HORIZON_DAYS = 90
# Origins, not rows, are the unit of the forward split: one origin date contributes
# one row per horizon, so a row-wise cut would put the same origin on both sides.
MIN_TRAIN_ORIGINS = 40


class ContractError(ValueError):
    """Raised when a frame does not satisfy the model contract."""


@dataclass
class ModelContract:
    """What the persisted pipeline expects and produces."""

    features: list[str]
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES
    target: str = DELTA_TARGET
    level_target: str = LEVEL_TARGET
    horizon_days: int = TARGET_HORIZON_DAYS
    requires_date: bool = True
    # Column that carries the point-in-time anchor: `date` for the weekly frame,
    # `origin_date` for the forward frame (keyed by origin_date + target_date +
    # forecast_horizon_days). Validation and splitting both use this, never a
    # hard-coded "date".
    date_column: str = "date"
    season_vocabulary: tuple[str, ...] = SEASON_VOCABULARY
    transformer: str = "seasonal_index.ExpandingSeasonalIndex"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "features": list(self.features),
            "categorical_features": list(self.categorical_features),
            "target": self.target,
            "level_target": self.level_target,
            "horizon_days": self.horizon_days,
            "requires_date": self.requires_date,
            "date_column": self.date_column,
            "season_vocabulary": list(self.season_vocabulary),
            "transformer": self.transformer,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelContract":
        return cls(
            features=[str(c) for c in payload["features"]],
            categorical_features=tuple(payload.get("categorical_features", CATEGORICAL_FEATURES)),
            target=payload.get("target", LEVEL_TARGET),
            level_target=payload.get("level_target", LEVEL_TARGET),
            horizon_days=int(payload.get("horizon_days", TARGET_HORIZON_DAYS)),
            requires_date=bool(payload.get("requires_date", True)),
            # `to_dict` writes date_column, so `from_dict` has to read it back:
            # the forward artifact anchors on `origin_date`, and a contract that
            # round-trips to "date" makes the serving app reject a perfectly good
            # feature vector.
            date_column=str(payload.get("date_column", "date")),
            season_vocabulary=tuple(payload.get("season_vocabulary", SEASON_VOCABULARY)),
            transformer=payload.get("transformer", "unknown"),
            extra=payload.get("extra", {}) or {},
        )


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def chronological_split(
    frame: pd.DataFrame,
    test_size: float,
    horizon_days: int = TARGET_HORIZON_DAYS,
    date_col: str = "date",
    min_test_rows: int = MIN_TEST_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split a frame chronologically and embargo the boundary.

    A row at date t carries a label for t + horizon_days. Cutting at position
    `cut` therefore leaks the labels of the last `horizon_days` training rows into
    the first test rows. The embargo drops those rows from the test side.

    Returns (train, test, audit); the audit dict carries the boundary dates so the
    split can be reproduced and audited from the logged run alone.
    """
    if date_col not in frame.columns:
        raise ContractError(f"'{date_col}' is required for a chronological split")
    if not 0.0 < test_size < 1.0:
        raise ContractError(f"test_size must be in (0, 1), got {test_size}")
    if horizon_days < 0:
        raise ContractError(f"horizon_days must be >= 0, got {horizon_days}")

    frame = frame.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(frame[date_col])
    if dates.duplicated().any():
        dup = dates[dates.duplicated()].dt.date.unique()[:5]
        raise ContractError(f"duplicate {date_col} values in the frame, e.g. {dup}")

    n = len(frame)
    cut = int(n * (1.0 - test_size))
    embargo_end = cut + horizon_days

    if cut < MIN_TRAIN_ROWS:
        raise ContractError(f"train side too small after split: {cut} rows < {MIN_TRAIN_ROWS}")
    test_rows = n - embargo_end
    if test_rows < min_test_rows:
        raise ContractError(
            f"test side too small after the {horizon_days}-day embargo: "
            f"{test_rows} rows < {min_test_rows}"
        )

    train = frame.iloc[:cut]
    test = frame.iloc[embargo_end:]

    audit = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "dropped_by_embargo": int(embargo_end - cut),
        "train_start_date": str(dates.iloc[0].date()),
        "train_end_date": str(dates.iloc[cut - 1].date()),
        "train_label_end_date": str(
            (dates.iloc[cut - 1] + pd.Timedelta(days=horizon_days)).date()
        ),
        "test_start_date": str(dates.iloc[embargo_end].date()),
        "test_end_date": str(dates.iloc[-1].date()),
        "embargo_days": int(horizon_days),
        "horizon_days": int(horizon_days),
    }

    # Hard invariant: no training label window may touch a test feature row.
    if pd.Timestamp(audit["train_label_end_date"]) >= pd.Timestamp(audit["test_start_date"]):
        raise ContractError(
            f"split leaks: train labels reach {audit['train_label_end_date']} but "
            f"test features start {audit['test_start_date']}"
        )
    return train, test, audit


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_features(frame: pd.DataFrame, contract: ModelContract) -> None:
    """Fail loudly when a frame does not match the persisted model contract."""
    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"expected a pandas DataFrame, got {type(frame).__name__}")

    missing = [c for c in contract.features if c not in frame.columns]
    if missing:
        raise ContractError(f"frame is missing model features: {missing}")

    if contract.requires_date and contract.date_column not in frame.columns:
        raise ContractError(
            f"frame is missing '{contract.date_column}'; the pipeline derives the "
            f"point-in-time features and the chronological split from it"
        )

    for col in contract.categorical_features:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            raise ContractError(
                f"'{col}' arrived integer-coded ({frame[col].dtype}); the persisted "
                f"OneHotEncoder expects the raw category strings "
                f"{list(contract.season_vocabulary)}"
            )

    # Every categorical feature of the contract is vocabulary-checked, not just a
    # hard-coded "season": the forward model carries the same four seasons under
    # the name `target_season`, and an unseen value there would be encoded as an
    # all-zero one-hot block exactly like in the weekly model.
    for col in contract.categorical_features:
        if col not in frame.columns:
            continue
        seen = {str(v) for v in frame[col].dropna()}
        unknown = seen - set(contract.season_vocabulary)
        if unknown:
            raise ContractError(
                f"unseen {col} values {sorted(unknown)}; the training vocabulary is "
                f"{list(contract.season_vocabulary)}. OneHotEncoder(handle_unknown="
                f"'ignore') would silently encode them as an all-zero block."
            )


def persistence_baselines(test: pd.DataFrame, level_target: str = LEVEL_TARGET) -> dict[str, float]:
    """Reference MSEs on the price level that the model must beat.

    `current_price` is the operational baseline (next week's price is usually
    today's price); `price_mean_7d` is the smoother, weaker variant;
    `constant_delta` is current_price plus the mean training delta.
    """
    truth = test[level_target].to_numpy(dtype="float64")
    out: dict[str, float] = {}
    for name in ("current_price", "price_mean_7d"):
        if name in test.columns:
            pred = test[name].to_numpy(dtype="float64")
            out[name] = float(np.mean((truth - pred) ** 2))
    if DELTA_TARGET in test.columns and "current_price" in test.columns:
        const = float(test[DELTA_TARGET].mean())
        base = test["current_price"].to_numpy(dtype="float64") + const
        out["constant_delta"] = float(np.mean((truth - base) ** 2))
    return out


# --------------------------------------------------------------------------
# Forward (multi-horizon) splitting and baselines
# --------------------------------------------------------------------------
def forward_chronological_split(
    frame: pd.DataFrame,
    test_size: float,
    horizon_days: int = TARGET_HORIZON_DAYS,
    max_forecast_horizon_days: int = FORWARD_MAX_HORIZON_DAYS,
    origin_col: str = "origin_date",
    horizon_col: str = "forecast_horizon_days",
    min_test_rows: int = MIN_TEST_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split the forward frame by *origin date* and embargo the boundary.

    A forward row (origin o, horizon h) labels the week after o + h, so its label
    window ends at o + h + horizon_days. All horizons of one origin share the same
    feature values, which is why the cut is made between origin dates and not
    between rows: a row-wise cut would put the same origin - and therefore the
    same information - on both sides.

    The embargo then drops every training row whose label window reaches the first
    test origin date. With a 90-day horizon that is a wide band; it is the price of
    a leak-free multi-horizon evaluation.

    Returns (train, test, audit).
    """
    for col in (origin_col, horizon_col):
        if col not in frame.columns:
            raise ContractError(f"'{col}' is required for a forward chronological split")
    if not 0.0 < test_size < 1.0:
        raise ContractError(f"test_size must be in (0, 1), got {test_size}")
    if horizon_days < 0 or max_forecast_horizon_days < 0:
        raise ContractError(
            f"horizon_days and max_forecast_horizon_days must be >= 0, got "
            f"{horizon_days} / {max_forecast_horizon_days}"
        )

    frame = frame.sort_values([origin_col, horizon_col]).reset_index(drop=True)
    origins = pd.to_datetime(frame[origin_col])
    unique_origins = pd.DatetimeIndex(sorted(origins.unique()))

    cut = int(len(unique_origins) * (1.0 - test_size))
    if cut < MIN_TRAIN_ORIGINS:
        raise ContractError(
            f"train side too small after split: {cut} origins < {MIN_TRAIN_ORIGINS}"
        )
    if len(unique_origins) - cut < 1:
        raise ContractError("split leaves no origin dates on the test side")

    test_start = pd.Timestamp(unique_origins[cut])
    train_mask = origins < test_start
    test_mask = origins >= test_start

    # A training row is only usable if its whole label window closes before the
    # first test origin date.
    label_end = (
        origins
        + pd.to_timedelta(frame[horizon_col].astype("int64"), unit="D")
        + pd.Timedelta(days=horizon_days)
    )
    leak_mask = train_mask & (label_end >= test_start)
    kept_train = train_mask & ~leak_mask

    train = frame.loc[kept_train]
    test = frame.loc[test_mask]

    if len(train) < MIN_TRAIN_ROWS:
        raise ContractError(
            f"train side too small after the embargo: {len(train)} rows < {MIN_TRAIN_ROWS}"
        )
    if len(test) < min_test_rows:
        raise ContractError(f"test side too small: {len(test)} rows < {min_test_rows}")

    audit = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "dropped_by_embargo": int(train_mask.sum()) - len(train),
        "train_origins": int(train_mask.sum()),
        "test_origins": int(test_mask.sum()),
        "train_start_date": str(origins.min().date()),
        "train_end_date": str(origins[train_mask].max().date()),
        "train_label_end_date": str(label_end[kept_train].max().date()),
        "test_start_date": str(test_start.date()),
        "test_end_date": str(origins.max().date()),
        "embargo_days": int(horizon_days + max_forecast_horizon_days),
        "horizon_days": int(horizon_days),
        "max_forecast_horizon_days": int(max_forecast_horizon_days),
    }

    # Hard invariant: no training label window may touch a test feature row.
    if pd.Timestamp(audit["train_label_end_date"]) >= pd.Timestamp(audit["test_start_date"]):
        raise ContractError(
            f"split leaks: train labels reach {audit['train_label_end_date']} but "
            f"test features start {audit['test_start_date']}"
        )
    return train, test, audit

def forward_persistence_baselines(
    test: pd.DataFrame,
    level_target: str = FORWARD_LEVEL_TARGET,
    origin_price_col: str = FORWARD_ORIGIN_PRICE,
    delta_target: str = FORWARD_DELTA_TARGET,
) -> dict[str, float]:
    """Reference MSEs a forward forecast must beat.

    `origin_current_price` is the operational baseline: the last price known when
    the forecast is made. `origin_price_mean_7d` is the smoother, weaker variant.
    `constant_delta` is the origin price plus the mean training delta - the best a
    model with no usable signal can do, which is why a model that only learns the
    average drift must still be measured against it.
    """
    if level_target not in test.columns:
        raise ContractError(f"'{level_target}' missing from the evaluation frame")
    truth = test[level_target].to_numpy(dtype="float64")
    out: dict[str, float] = {}
    for name in (origin_price_col, "origin_price_mean_7d"):
        if name in test.columns:
            pred = test[name].to_numpy(dtype="float64")
            out[name] = float(np.mean((truth - pred) ** 2))
    if delta_target in test.columns and origin_price_col in test.columns:
        const = float(test[delta_target].mean())
        base = test[origin_price_col].to_numpy(dtype="float64") + const
        out["constant_delta"] = float(np.mean((truth - base) ** 2))
    return out


def forward_contract(features: list[str], **extra) -> ModelContract:
    """The contract of the forward artifact, built from the shared constants.

    `date_column` is `origin_date`: the forward frame has no `date`, and the split
    must anchor on the date the forecast was made, not on the date it forecasts.

    `requires_date` is False because the anchor is not part of the model matrix.
    The weekly model needs `date` inside the pipeline (ExpandingSeasonalIndex
    computes the seasonal index at predict time); the forward model gets its
    point-in-time values precomputed in the feature view, and `origin_date` is a
    lookup key that must never become a feature.
    """
    return ModelContract(
        features=list(features),
        categorical_features=FORWARD_CATEGORICAL_FEATURES,
        target=FORWARD_DELTA_TARGET,
        level_target=FORWARD_LEVEL_TARGET,
        horizon_days=TARGET_HORIZON_DAYS,
        requires_date=False,
        date_column="origin_date",
        transformer="preprocessing.ColumnTransformer(OneHotEncoder)",
        extra=dict(extra),
    )

