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
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd

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
from model_contract import (
    DELTA_TARGET,
    LEVEL_TARGET,
    MAX_SKILL_VS_PERSISTENCE,
    MAX_SKILL_VS_PERSISTENCE_TOLERANCE,
    MIN_DRIVER_COVERAGE,
    TARGET_HORIZON_DAYS,
    ContractError,
    ModelContract,
    chronological_split,
)
from training_pipeline import (
    build_model,
    evaluate,
    get_s3_options,
    save_contract_to_s3,
    save_model_to_s3,
)


def encode_features(X: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - removed
    """Deprecated: integer-coding categoricals is no longer part of the contract.

    The persisted pipeline one-hot encodes `season` from the raw strings, so the
    vocabulary is keyed by category name. Integer coding here produced a
    position-keyed model that silently degraded whenever the serving side coded
    "Frühling" differently. Kept only so stale imports fail loudly at the call
    site rather than producing a wrong model.
    """
    raise ContractError(
        "encode_features() was removed: pass raw category strings to build_model()"
    )



def register_model(project, model, mse: float, features: list | None = None,
                   contract: ModelContract | None = None):
    """Register the model in the Hopsworks Model Registry.

    Hopsworks 5.x model registry: `Model.save(model_folder)` uploads a folder
    of model files as the model artifact. With `id=None` a new model is
    created, and with `version=None` the next version number is assigned
    automatically (1, 2, 3, ...).

    `features` is written as a sidecar file so the inference app knows exactly
    which feature-view columns this model expects (see src/model_artifact.py).
    The full `contract` (target, horizon, categorical vocabulary) travels next to
    the pickle so serving can validate a request before calling predict().
    """
    from hsml.model import Model

    with tempfile.TemporaryDirectory(prefix="hw_model_") as tmp:
        model_dir = Path(tmp) / MODEL_NAME
        model_dir.mkdir()
        joblib.dump(model, model_dir / MODEL_FILE_NAME)
        if features:
            write_model_features(model_dir, MODEL_NAME, features, contract)

        registry = project.get_model_registry()
        reg_model = Model(
            id=None,
            name=MODEL_NAME,
            description="RandomForest pipeline -> target_week_change (heizoel24, weekly)",
            framework="SKLEARN",
            metrics={"level_mse": float(mse)},
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
    X_split_train, X_split_test, y_split_train, y_split_test = fv.train_test_split(
        test_size=TEST_SIZE
    )

    # The feature group also holds the most recent dates whose 7-day-ahead
    # label is NULL (stored for serving, see hw_feature_pipeline.py); rows
    # without a label are useless for training and are dropped here.
    def _drop_unlabeled(X: pd.DataFrame, y) -> tuple:
        y = pd.DataFrame(y)
        keep = y.iloc[:, 0].notna().to_numpy()
        return X[keep], y[keep]

    n_total = len(y_split_train) + len(y_split_test)
    X_split_train, y_split_train = _drop_unlabeled(X_split_train, y_split_train)
    X_split_test, y_split_test = _drop_unlabeled(X_split_test, y_split_test)
    print(f"feature view rows: {n_total}, dropped {n_total - len(y_split_train) - len(y_split_test)} rows without a label")

    # 2) Model feature columns: the config says which drivers should be used,
    #    the feature view says what is actually available. Driver columns that
    #    the view does not have (yet) are reported and skipped instead of
    #    failing the training.
    wanted = weekly_model_features()
    missing = [col for col in wanted if col not in X_split_train.columns]
    if missing:
        print(f"WARNING: feature view '{FEATURE_VIEW_NAME}' v{FEATURE_VIEW_VERSION} lacks {missing}")
    model_features = [col for col in wanted if col in X_split_train.columns]
    driver_features = [col for col in model_features if col not in FEATURES]
    if "date" not in X_split_train.columns:
        raise ContractError(
            f"feature view '{FEATURE_VIEW_NAME}' does not expose 'date'; the split "
            f"and the point-in-time seasonal index both require it"
        )
    model_inputs = model_features + (["date"] if "date" not in model_features else [])

    # 3) Chronological split with embargo. fv.train_test_split() shuffles, which
    #    leaks the future into training; the frames are recombined and re-split by
    #    date so the embargo is applied to post-dropna positions.
    frame = pd.concat([X_split_train, X_split_test], ignore_index=True)
    frame[LEVEL_TARGET] = pd.concat(
        [pd.DataFrame(y_split_train), pd.DataFrame(y_split_test)], ignore_index=True
    ).iloc[:, 0].to_numpy()

    # Core features must be complete; DRIVER columns may be NULL. A driver is
    # NULL on a day when its source did not publish (the market series exists on
    # trading days only) or before the series started, and the feature pipeline
    # already carried the last past value for up to DRIVER_CARRY_DAYS days.
    # Requiring every driver non-null - what this pipeline used to do - deleted
    # 81 % of the labelled rows (8 659 -> 1 615), left the forest 1 292 training
    # rows and was the main reason the weekly model lost to persistence. The
    # forward pipeline has treated drivers as optional all along.
    core_features = [col for col in model_features if col not in driver_features]
    frame = frame.dropna(subset=core_features + [LEVEL_TARGET]).sort_values("date").reset_index(
        drop=True
    )

    # A driver present on only a sliver of the history is not a driver, it is a
    # column of mostly-NULL values the forest will split on. What decides that is
    # coverage WHERE THE MODEL IS JUDGED, not over the whole history: the Rhine
    # series starts 2020-06 while the price history starts 2003, so a driver can
    # be NULL on 73 % of all rows and still be present on every hold-out day.
    # Dropping it for that would throw away the only physical signal the model
    # has; keeping a driver that is absent in the hold-out is the mistake this
    # guard exists for (Kaub: WSV publishes ~30 days).
    unusable = []
    holdout = frame.tail(max(1, int(len(frame) * TEST_SIZE)))
    for col in driver_features:
        overall = float(frame[col].notna().mean()) if len(frame) else 0.0
        in_holdout = float(holdout[col].notna().mean()) if len(holdout) else 0.0
        if in_holdout < MIN_DRIVER_COVERAGE:
            unusable.append((col, overall, in_holdout))
    if unusable:
        print(
            "driver features dropped for missing coverage in the hold-out "
            f"(< {MIN_DRIVER_COVERAGE:.0%}): "
            + ", ".join(f"{col}=all {all_share:.1%}/hold-out {hold_share:.1%}"
                        for col, all_share, hold_share in unusable)
        )
        model_features = [col for col in model_features if col not in {c for c, _, _ in unusable}]
        driver_features = [col for col in model_features if col not in core_features]
    model_inputs = model_features + (["date"] if "date" not in model_features else [])

    frame[DELTA_TARGET] = frame[LEVEL_TARGET] - frame["current_price"]

    train, test, audit = chronological_split(
        frame, test_size=TEST_SIZE, horizon_days=TARGET_HORIZON_DAYS
    )
    X_train, y_train = train[model_inputs], train[DELTA_TARGET]
    print(f"X_train: {X_train.shape} | y_train (delta): {y_train.shape}")
    print(
        f"train {audit['train_start_date']} .. {audit['train_end_date']} "
        f"(labels reach {audit['train_label_end_date']}) | "
        f"test {audit['test_start_date']} .. {audit['test_end_date']} | "
        f"embargo {audit['embargo_days']}d, {audit['dropped_by_embargo']} rows dropped"
    )
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
        metrics = evaluate(model, train, test, input_columns=model_inputs)

        for key, value in audit.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_type", "Pipeline(ExpandingSeasonalIndex + ColumnTransformer + RandomForestRegressor)")
        mlflow.log_param("target", DELTA_TARGET)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("data_source", f"hopsworks feature view {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}")
        mlflow.log_param("n_features", len(model_features))
        mlflow.log_param("driver_features", ",".join(driver_features) or "none")
        mlflow.log_metric("level_mse", metrics["level_mse"])
        mlflow.log_metric("level_r2", metrics["level_r2"])
        mlflow.log_metric("delta_mse", metrics["delta_mse"])
        mlflow.log_metric("delta_r2", metrics["delta_r2"])
        for name, value in metrics["baselines"].items():
            mlflow.log_metric(f"baseline_level_mse_{name}", value)
        mlflow.log_metric("skill_vs_persistence", metrics["skill_vs_persistence"])

        # Contextual buckets: the same hold-out split by how much the price
        # actually moved. In a week that did not move, nothing beats "no change";
        # the signal a model adds lives in the weeks that did. Reporting only the
        # average hides which of the two the model is failing on.
        buckets = metrics["buckets"]
        for name, bucket in buckets.items():
            mlflow.log_metric(f"level_mse_{name}", bucket["level_mse"])
            mlflow.log_metric(f"persistence_mse_{name}", bucket["persistence_mse"])
            mlflow.log_metric(f"skill_vs_persistence_{name}", bucket["skill"])

        overall = metrics["skill_vs_persistence"]
        volatile_skill = buckets.get("volatile", {}).get("skill", float("inf"))
        constant_delta = metrics["baselines"].get("constant_delta", float("inf"))

        strict_ok = overall < MAX_SKILL_VS_PERSISTENCE
        # Allowed only when the model is genuinely better where the market moves
        # AND better than the do-nothing constant-delta baseline overall - a
        # model that is merely close to persistence on average still fails.
        contextual_ok = (
            not strict_ok
            and overall <= MAX_SKILL_VS_PERSISTENCE_TOLERANCE
            and volatile_skill < MAX_SKILL_VS_PERSISTENCE
            and metrics["level_mse"] <= constant_delta
        )
        gate_ok = strict_ok or contextual_ok
        gate_rule = "strict" if strict_ok else ("contextual" if contextual_ok else "failed")

        mlflow.log_metric("promotion_gate", 1 if gate_ok else 0)
        mlflow.log_param("promotion_gate_rule", gate_rule)
        mlflow.log_metric("skill_vs_persistence_volatile", volatile_skill)

    mse = metrics["level_mse"]
    print(f"Level MSE: {mse:.2f} (R2 {metrics['level_r2']:.3f}) | "
          f"skill vs persistence {metrics['skill_vs_persistence']:.3f}")
    for name, bucket in metrics["buckets"].items():
        print(
            f"  {name:8s}: {bucket['rows']:5d} rows, MSE {bucket['level_mse']:7.2f} "
            f"vs persistence {bucket['persistence_mse']:7.2f} (skill {bucket['skill']:.3f})"
        )

    # 5) Promotion gate: never register or export a model that is not better than
    #    'next week's price is today's price' - either overall (strict) or where
    #    the market actually moves (contextual, see model_contract).
    if not gate_ok:
        print(
            "PROMOTION BLOCKED: not better than persistence overall "
            f"(skill {metrics['skill_vs_persistence']:.3f}, tolerance "
            f"{MAX_SKILL_VS_PERSISTENCE_TOLERANCE}) and not better in the volatile "
            f"weeks (skill {volatile_skill:.3f}) or than the constant-delta baseline "
            f"({metrics['level_mse']:.2f} vs {constant_delta:.2f}); "
            "registry and S3 export skipped.",
            file=sys.stderr,
        )
        return 2
    if not strict_ok:
        print(
            "PROMOTED ON THE CONTEXTUAL RULE: overall skill "
            f"{metrics['skill_vs_persistence']:.3f} (<= "
            f"{MAX_SKILL_VS_PERSISTENCE_TOLERANCE}) but strictly better than "
            f"persistence in the volatile weeks ({volatile_skill:.3f})."
        )

    # 6) Register in the Hopsworks Model Registry (auto versioning)
    contract = ModelContract(
        features=model_inputs,
        target=DELTA_TARGET,
        level_target=LEVEL_TARGET,
        horizon_days=TARGET_HORIZON_DAYS,
        extra={"driver_features": driver_features, "n_estimators": N_ESTIMATORS},
    )
    registered = register_model(project, model, mse, model_inputs, contract)
    print(f"Registered '{MODEL_NAME}' (version {registered.version}) in the Hopsworks Model Registry")

    # 4) Optional: export to S3 for the legacy inference app
    #    (skipped gracefully when no cluster access is available). The contract
    #    sidecar goes with the pickle: without it the app cannot tell this delta
    #    model from a pre-contract level model.
    try:
        s3_options = get_s3_options()
        save_model_to_s3(model, s3_options)
        save_contract_to_s3(contract, s3_options)
    except Exception as e:
        print(f"Note: S3 export skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
