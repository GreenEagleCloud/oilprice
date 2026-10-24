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


def write_model_features(model_dir, model_name: str, features: list, contract=None) -> None:
    """Write <model_dir>/model_features.json for a registered model folder.

    `contract` is an optional src/model_contract.ModelContract. When present the
    sidecar carries the whole input/output contract, not just the column list.
    """
    if contract is not None:
        payload = contract.to_dict()
        payload["model_name"] = model_name
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

    Older artifacts only list `features`; those are upgraded to a contract with
    the legacy level target so serving keeps working until they are retrained.
    """
    from model_contract import LEVEL_TARGET, ModelContract

    payload = _read_payload(folder)
    if payload is None:
        return None
    if not payload.get("features"):
        return None
    payload.setdefault("target", LEVEL_TARGET)
    return ModelContract.from_dict(payload)


def _read_payload(folder) -> dict | None:
    path = Path(folder) / FEATURES_FILE_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
