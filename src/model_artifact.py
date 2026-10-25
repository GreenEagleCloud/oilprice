"""Feature-list sidecar for the models in the Hopsworks Model Registry.

A model artifact is just a joblib pickle; the inference app needs to know which
feature-view columns it was trained on, because the weekly/forward feature
views gained the Rhine and market driver columns (v2). The training pipelines
therefore write a small JSON file next to the pickle inside the registered
model folder, and hw_app.py reads it after `model.download()`.

Without the sidecar hw_app would have to guess the column set, and a model that
was registered before a feature-view change would be fed the wrong columns.
"""

import json
from pathlib import Path

FEATURES_FILE_NAME = "model_features.json"
# Written into the sidecar whenever a full ModelContract is stored with it. It is
# the only reliable way to tell a contract artifact from one that predates
# src/model_contract.py, because a legacy sidecar is indistinguishable by content:
# it just lists `features`.
CONTRACT_VERSION = 1


def write_model_features(model_dir, model_name: str, features: list, contract=None) -> None:
    """Write <model_dir>/model_features.json for a registered model folder.

    `contract` is an optional src/model_contract.ModelContract. When present the
    sidecar carries the whole input/output contract plus `contract_version`, the
    marker read_model_contract uses to tell a contract artifact from a legacy one.
    """
    if contract is not None:
        payload = contract.to_dict()
        payload["model_name"] = model_name
        payload["contract_version"] = CONTRACT_VERSION
    else:
        payload = {"model_name": model_name, "features": [str(name) for name in features]}
    Path(model_dir, FEATURES_FILE_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_model_features(folder) -> list | None:
    """Return the trained feature list of a downloaded model folder, if present."""
    payload = _read_payload(folder)
    if payload is None:
        return None
    features = payload.get("features")
    return [str(name) for name in features] if features else None


def read_model_contract(folder):
    """Return the full ModelContract of a downloaded model folder, if present.

    Returns None for an artifact whose sidecar lists only `features` - one that
    predates src/model_contract.py. That distinction is not cosmetic: a legacy
    artifact was fitted on integer-coded category strings, so the serving app must
    integer-code its vector and must treat the prediction as a price level.
    Inventing a contract for it claims the opposite - the app then hands the
    estimator raw strings and it dies on "could not convert string to float:
    'Herbst'".
    """
    return contract_from_payload(_read_payload(folder))


def contract_from_payload(payload):
    """Rebuild a ModelContract from a sidecar payload, None for a legacy sidecar.

    Shared by the Model Registry path (read_model_contract) and the S3 fallback,
    which reads the sidecar straight out of the bucket instead of a downloaded
    model folder.

    Two payloads count as a contract:

    * one marked with `contract_version` - every sidecar written today;
    * one that declares its own `target` - sidecars written before the marker
      existed, where the training pipeline passed a contract and the writer
      emitted the whole contract dict.

    A sidecar that lists only `features` is refused. Nothing in it says how the
    artifact was encoded, and guessing a target is what broke the weekly model:
    the app handed raw category strings to an estimator fitted on integer codes.
    """
    from model_contract import ModelContract

    if not payload or not payload.get("features"):
        return None
    if payload.get("contract_version") != CONTRACT_VERSION and not payload.get("target"):
        return None
    return ModelContract.from_dict(payload)


def sidecar_key(model_key: str) -> str:
    """Sidecar key that belongs to a model key in the model bucket.

    `oil_price_forward_model.pkl` -> `oil_price_forward_model_features.json`.
    """
    stem = model_key.rsplit(".", 1)[0]
    return f"{stem}_features.json"


def sidecar_keys(model_key: str) -> list[str]:
    """Sidecar keys that can belong to a model key in the model bucket.

    The weekly pipeline exports the plain `model_features.json`
    (src/training_pipeline.py) while the forward pipeline exports a key named
    after its pickle, so both spellings are candidates.
    """
    return [sidecar_key(model_key), FEATURES_FILE_NAME]


def _read_payload(folder) -> dict | None:
    path = Path(folder) / FEATURES_FILE_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
