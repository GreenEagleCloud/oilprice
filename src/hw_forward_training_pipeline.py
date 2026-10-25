"""Training pipeline for the forward oil-price forecast model.

This trains the model for the `heizoel24_forward_forecast` feature view.

What the model regresses
------------------------
Not the price level, but the change of the week-after price relative to the
price that was already known when the forecast was made:

    forward_target_week_change
        = target_price_week_after_target_date - origin_current_price

The served level is rebuilt at inference time as
`origin_current_price + prediction`, and `target_price_week_after_target_date`
remains the column every metric is scored on.

Why a delta and not the level
-----------------------------
A RandomForest only predicts values inside the range it has seen - it cannot
extrapolate a level. Trained on the level, a 90-day forecast degenerates into an
average of historical levels and loses to the baseline "the price today is still
the price in 90 days". The delta is roughly stationary, so one forest can serve
every horizon and `forecast_horizon_days` becomes the feature that says how far
the drift may reach. This is the decision the weekly pipeline already made with
`target_week_change`; the forward model had not been moved over yet.

Rows are keyed by (target_date, origin_date, forecast_horizon_days), so the split
is chronological over `origin_date` with an embargo of
forecast_horizon_days + 7 - see src/model_contract.forward_chronological_split.
"""

import json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
from model_artifact import CONTRACT_VERSION, write_model_features
from model_contract import (
    FORWARD_DELTA_TARGET,
    FORWARD_LEVEL_TARGET,
    FORWARD_MAX_HORIZON_DAYS,
    FORWARD_ORIGIN_PRICE,
    MAX_SKILL_VS_PERSISTENCE,
    TARGET_HORIZON_DAYS,
    ContractError,
    forward_chronological_split,
    forward_contract,
    forward_persistence_baselines,
    validate_features,
)
from training_pipeline import get_s3_options

# The feature-view label name and the contract level target must name the same
# column. FORWARD_TARGET is part of the Hopsworks feature-group schema, so it is
# not a free variable: renaming it here would train on a column the view does not
# have. If a future feature-group version renames the label, this check fails at
# import time instead of scoring a different column silently.
if FORWARD_TARGET != FORWARD_LEVEL_TARGET:
    raise ContractError(
        f"hw_config.FORWARD_TARGET ({FORWARD_TARGET!r}) and model_contract."
        f"FORWARD_LEVEL_TARGET ({FORWARD_LEVEL_TARGET!r}) disagree; the feature "
        f"view label and the model contract must be the same column"
    )

# Horizon buckets the report is broken down by: the front week is what the app
# serves most often, 61-90 days is where a level-trained model used to collapse.
HORIZON_BUCKETS = ((1, 7), (8, 30), (31, 60), (61, FORWARD_MAX_HORIZON_DAYS))


def add_forward_delta(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialise the stationary forward target from the stored level label.

    Mirrors the weekly `target_week_change = target_next_week_price -
    current_price`: the model learns the movement away from a price that is
    already known, never the level itself.
    """
    for col in (FORWARD_LEVEL_TARGET, FORWARD_ORIGIN_PRICE):
        if col not in frame.columns:
            raise ContractError(
                f"'{col}' is missing from the forward frame; the feature view "
                f"'{FORWARD_FEATURE_VIEW_NAME}' must expose the level label and "
                f"the origin price the delta is measured against"
            )
    out = frame.copy()
    out[FORWARD_DELTA_TARGET] = (
        out[FORWARD_LEVEL_TARGET].astype("float64") - out[FORWARD_ORIGIN_PRICE].astype("float64")
    )
    return out


def _drop_unlabeled(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows whose forward label is not known yet (the future serving rows)."""
    keep = frame[FORWARD_LEVEL_TARGET].notna() & frame[FORWARD_DELTA_TARGET].notna()
    return frame.loc[keep].reset_index(drop=True), int((~keep).sum())


def build_forward_model(features: list[str], categorical: tuple[str, ...] = ("target_season",)):
    """Build the compact RandomForest pipeline of the forward feature view.

    The categorical block is named by the contract instead of being sniffed from
    the frame dtypes: an artifact whose one-hot block was fitted on an
    integer-coded column cannot be served from raw category strings, and sniffing
    is how that mismatch was created in the first place.
    """
    missing = [col for col in categorical if col not in features]
    if missing:
        raise ContractError(f"categorical features not in the model: {missing}")
    numeric_cols = [col for col in features if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
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


def evaluate_forward_model(
    model, train: pd.DataFrame, test: pd.DataFrame, input_columns: list[str]
) -> dict:
    """Score the delta model on the price level the app will be judged on.

    The model outputs `forward_target_week_change`; a forecast is only useful as a
    price, so every headline metric reconstructs
    `origin_current_price + predicted_delta` and compares it with persistence.
    Reporting delta-MSE alone would hide that the model is worse than "the price
    stays where it is".
    """
    predicted_delta = model.predict(test[input_columns])
    origin = test[FORWARD_ORIGIN_PRICE].to_numpy(dtype="float64")
    truth_level = test[FORWARD_LEVEL_TARGET].to_numpy(dtype="float64")
    level_pred = origin + predicted_delta

    level_mse = float(mean_squared_error(truth_level, level_pred))
    level_mae = float(mean_absolute_error(truth_level, level_pred))
    level_r2 = float(r2_score(truth_level, level_pred))
    delta_mse = float(mean_squared_error(test[FORWARD_DELTA_TARGET], predicted_delta))
    delta_r2 = float(r2_score(test[FORWARD_DELTA_TARGET], predicted_delta))

    baselines = forward_persistence_baselines(test)
    reference = baselines.get(FORWARD_ORIGIN_PRICE, float("nan"))
    skill = level_mse / reference if reference else float("inf")

    # Train-side sanity: the forest must at least fit the window it was trained on.
    train_delta = model.predict(train[input_columns])
    train_level = train[FORWARD_ORIGIN_PRICE].to_numpy(dtype="float64") + train_delta
    train_level_mse = float(mean_squared_error(train[FORWARD_LEVEL_TARGET], train_level))

    horizons = test["forecast_horizon_days"].to_numpy(dtype="int64")
    by_horizon = {}
    for low, high in HORIZON_BUCKETS:
        mask = (horizons >= low) & (horizons <= high)
        if not mask.any():
            continue
        key = f"{low}_{high}d"
        by_horizon[key] = {
            "rows": int(mask.sum()),
            "level_mse": float(mean_squared_error(truth_level[mask], level_pred[mask])),
            "baseline_mse": float(mean_squared_error(truth_level[mask], origin[mask])),
        }
        base = by_horizon[key]["baseline_mse"]
        by_horizon[key]["skill_vs_persistence"] = (
            by_horizon[key]["level_mse"] / base if base else float("inf")
        )

    return {
        "level_mse": level_mse,
        "level_mae": level_mae,
        "level_r2": level_r2,
        "delta_mse": delta_mse,
        "delta_r2": delta_r2,
        "train_level_mse": train_level_mse,
        "baselines": baselines,
        "skill_vs_persistence": skill,
        "by_horizon": by_horizon,
    }



def register_forward_model(project, model, metrics: dict, features: list, contract) -> Model:
    """Register the forward model in the Hopsworks Model Registry.

    The sidecar written next to the pickle is the full ModelContract, so the
    inference app can tell whether this artifact predicts a delta or a level
    instead of guessing from the file name.
    """
    registry = project.get_model_registry()

    with tempfile.TemporaryDirectory(prefix="hw_forward_model_") as tmp:
        model_dir = Path(tmp) / FORWARD_MODEL_NAME
        model_dir.mkdir()
        joblib.dump(model, model_dir / FORWARD_MODEL_FILE_NAME)
        write_model_features(model_dir, FORWARD_MODEL_NAME, features, contract)

        reg_model = Model(
            id=None,
            name=FORWARD_MODEL_NAME,
            description=(
                "RandomForest pipeline -> forward_target_week_change, served as "
                "origin_current_price + delta against "
                "target_price_week_after_target_date (heizoel24 forward forecast "
                "with forecast_horizon_days)"
            ),
            framework="SKLEARN",
            metrics={
                "level_mse": float(metrics["level_mse"]),
                "delta_mse": float(metrics["delta_mse"]),
                "skill_vs_persistence": float(metrics["skill_vs_persistence"]),
            },
            model_registry_id=registry.model_registry_id,
        )
        return reg_model.save(str(model_dir))


def save_forward_model_to_s3(model, s3_options: dict, contract=None) -> None:
    """Export the forward pickle plus its contract sidecar to the model bucket.

    hw_app.py falls back to S3 when the Model Registry is unreachable, and there
    it has no downloaded folder to read the sidecar from - so the sidecar has to
    sit next to the pickle in the bucket, otherwise a delta model would be served
    as if it predicted a price level.
    """
    import s3fs

    from model_artifact import sidecar_key

    s3 = s3fs.S3FileSystem(**s3_options)
    path = f"s3://{MODEL_BUCKET}/{FORWARD_MODEL_FILE_NAME}"
    with s3.open(path, "wb") as f:
        joblib.dump(model, f)
    print(f"Forward model saved to {path}")

    if contract is None:
        return
    payload = contract.to_dict()
    payload["model_name"] = FORWARD_MODEL_NAME
    payload["contract_version"] = CONTRACT_VERSION
    sidecar = f"s3://{MODEL_BUCKET}/{sidecar_key(FORWARD_MODEL_FILE_NAME)}"
    with s3.open(sidecar, "wb") as f:
        f.write(json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
    print(f"Forward contract saved to {sidecar}")



def main() -> int:
    try:
        project = login()
        fs = project.get_feature_store()
        fv = fs.get_feature_view(
            FORWARD_FEATURE_VIEW_NAME, version=FORWARD_FEATURE_VIEW_VERSION
        )

        # fv.train_test_split() shuffles. In a forward view that is wrong twice
        # over: every horizon of one origin_date carries the SAME feature values,
        # and a 90-day training row carries a label three months past its origin.
        # Both halves are recombined and re-split chronologically over origin_date.
        X_split_train, X_split_test, y_split_train, y_split_test = fv.train_test_split(
            test_size=TEST_SIZE
        )
        frame = pd.concat([X_split_train, X_split_test], ignore_index=True)
        frame[FORWARD_LEVEL_TARGET] = pd.concat(
            [pd.DataFrame(y_split_train), pd.DataFrame(y_split_test)], ignore_index=True
        ).iloc[:, 0].to_numpy()

        # target_date is part of the online lookup key but says nothing the model
        # may see - origin_date + forecast_horizon_days already determine it - so
        # it is dropped rather than risk becoming a feature.
        frame = frame.drop(columns=[c for c in ("target_date",) if c in frame.columns])
        for col in ("origin_date", "forecast_horizon_days"):
            if col not in frame.columns:
                raise ContractError(
                    f"forward feature view lacks '{col}'; the chronological split "
                    f"and the horizon feature both need it"
                )

        wanted = forward_model_features()
        missing_core = [c for c in FORWARD_FEATURES if c not in frame.columns]
        if missing_core:
            raise ContractError(f"forward feature view is missing features: {missing_core}")

        # Driver columns are optional: a forward view created before the Rhine /
        # market join simply lacks them, and the model is then trained without.
        missing_drivers = [c for c in wanted if c not in frame.columns]
        if missing_drivers:
            print(f"WARNING: forward feature view lacks driver features: {missing_drivers}")
        model_features = [c for c in wanted if c in frame.columns]
        driver_features = [c for c in model_features if c not in FORWARD_FEATURES]
        print(f"driver features used: {len(driver_features)} ({', '.join(driver_features)})")

        # Core features must be complete. Drivers may legitimately be NULL - the
        # Rhine series starts in 2020 and the market series only has trading days
        # - and a RandomForest splits on missing values instead of deleting the
        # whole origin, which is what a blanket dropna would do. The label is NOT
        # dropped here: the rows the feature view stores for future serving dates
        # have no label yet, and _drop_unlabeled reports how many of them there are.
        required = [c for c in model_features if c not in driver_features]
        frame = frame.dropna(subset=required)
        frame = add_forward_delta(frame)
        frame, dropped = _drop_unlabeled(frame)
        print(f"forward rows with a label: {len(frame)} (dropped {dropped} without a label)")

        train, test, audit = forward_chronological_split(
            frame,
            test_size=TEST_SIZE,
            horizon_days=TARGET_HORIZON_DAYS,
            max_forecast_horizon_days=int(frame["forecast_horizon_days"].max()),
        )
        X_train, y_train = train[model_features], train[FORWARD_DELTA_TARGET]
        X_test = test[model_features]

        contract = forward_contract(
            model_features,
            driver_features=driver_features,
            n_estimators=FORWARD_N_ESTIMATORS,
            max_forecast_horizon_days=audit["max_forecast_horizon_days"],
        )
        validate_features(X_train, contract)
        validate_features(X_test, contract)

        print(
            f"train {audit['train_rows']} rows {audit['train_start_date']} .. "
            f"{audit['train_end_date']} (labels reach {audit['train_label_end_date']}) | "
            f"test {audit['test_rows']} rows {audit['test_start_date']} .. "
            f"{audit['test_end_date']} | embargo {audit['embargo_days']}d, "
            f"{audit['dropped_by_embargo']} rows dropped"
        )

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI_DEFAULT
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        # Autolog must be active BEFORE model.fit(): it logs the model exactly once
        # (name "model") together with the training metrics.
        mlflow.autolog()

        model = build_forward_model(model_features, contract.categorical_features)
        with mlflow.start_run():
            model.fit(X_train, y_train)
            metrics = evaluate_forward_model(model, train, test, input_columns=model_features)

            for key, value in audit.items():
                mlflow.log_param(key, value)
            mlflow.log_param("model_type", "Pipeline(ColumnTransformer + RandomForestRegressor)")
            mlflow.log_param("model_variant", "forward_forecast")
            mlflow.log_param("target", FORWARD_DELTA_TARGET)
            mlflow.log_param("level_target", FORWARD_LEVEL_TARGET)
            mlflow.log_param("n_estimators", FORWARD_N_ESTIMATORS)
            mlflow.log_param("max_depth", FORWARD_MAX_DEPTH)
            mlflow.log_param("min_samples_leaf", FORWARD_MIN_SAMPLES_LEAF)
            mlflow.log_param("max_features", FORWARD_MAX_FEATURES)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param(
                "data_source",
                f"hopsworks feature view {FORWARD_FEATURE_VIEW_NAME} "
                f"v{FORWARD_FEATURE_VIEW_VERSION}",
            )
            mlflow.log_param("n_features", len(model_features))
            mlflow.log_param("driver_features", ",".join(driver_features) or "none")

            mlflow.log_metric("level_mse", metrics["level_mse"])
            mlflow.log_metric("level_mae", metrics["level_mae"])
            mlflow.log_metric("level_r2", metrics["level_r2"])
            mlflow.log_metric("delta_mse", metrics["delta_mse"])
            mlflow.log_metric("delta_r2", metrics["delta_r2"])
            mlflow.log_metric("train_level_mse", metrics["train_level_mse"])
            for name, value in metrics["baselines"].items():
                mlflow.log_metric(f"baseline_level_mse_{name}", value)
            for key, values in metrics["by_horizon"].items():
                mlflow.log_metric(f"level_mse_{key}", values["level_mse"])
                mlflow.log_metric(f"baseline_mse_{key}", values["baseline_mse"])
                mlflow.log_metric(f"skill_{key}", values["skill_vs_persistence"])
            mlflow.log_metric("skill_vs_persistence", metrics["skill_vs_persistence"])
            gate_ok = metrics["skill_vs_persistence"] < MAX_SKILL_VS_PERSISTENCE
            mlflow.log_metric("promotion_gate", 1 if gate_ok else 0)

        print(
            f"Level MSE: {metrics['level_mse']:.2f} (MAE {metrics['level_mae']:.2f}, "
            f"R2 {metrics['level_r2']:.3f}) | delta MSE: {metrics['delta_mse']:.2f}"
        )
        print(
            f"Skill vs persistence: {metrics['skill_vs_persistence']:.3f} "
            f"(< {MAX_SKILL_VS_PERSISTENCE} required)"
        )
        for key, values in metrics["by_horizon"].items():
            print(
                f"  {key}: {values['rows']} rows, MSE {values['level_mse']:.2f} vs "
                f"persistence {values['baseline_mse']:.2f} "
                f"(skill {values['skill_vs_persistence']:.3f})"
            )

        # Promotion gate: a forward model that is not better than "the price stays
        # where it is" must not reach the registry or S3. The inference app then
        # keeps serving the previous version instead of a worse one.
        if not gate_ok:
            print(
                "PROMOTION BLOCKED: the forward model is not better than persistence "
                "on the chronological hold-out; registry and S3 export skipped.",
                file=sys.stderr,
            )
            return 2

        registered = register_forward_model(project, model, metrics, model_features, contract)
        print(
            f"Registered '{FORWARD_MODEL_NAME}' (version {registered.version}) in the "
            "Hopsworks Model Registry"
        )

        try:
            s3_options = get_s3_options()
            save_forward_model_to_s3(model, s3_options, contract)
        except Exception as e:
            print(f"Note: S3 export skipped: {e}")

        return 0
    except ContractError as e:
        print(f"CONTRACT ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
