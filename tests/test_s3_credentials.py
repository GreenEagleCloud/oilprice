"""S3 credential wiring: options come from the environment, never from kubectl.

The training Job runs with `allowPrivilegeEscalation: false`, `capabilities:
drop: [ALL]` and `runAsNonRoot`. Shelling out to `kubectl get secret
rook-ceph-object-user-...` would require a kubeconfig inside the pod plus RBAC
that may read Secrets in the *rook-ceph* namespace - a cross-namespace secret
read that no training job should ever hold. These tests pin the replacement:
secretKeyRef env vars in, s3fs options out, and no subprocess anywhere.
"""

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import s3_credentials as sc  # noqa: E402
import training_pipeline as tp  # noqa: E402
import upload_train_data as utd  # noqa: E402

ENDPOINT = "http://rook-ceph-rgw-objectstore.rook-ceph.svc:8080"
ENV_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL", "AWS_ENDPOINT_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test may inherit real cluster credentials from the developer shell."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _inject(monkeypatch, endpoint_var: str = "AWS_ENDPOINT_URL") -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI-K7MDENG-bPxRfiCYEXAMPLEKEY")
    monkeypatch.setenv(endpoint_var, ENDPOINT)


# ------------------------------------------------------------- options ----
def test_options_are_read_from_the_environment(monkeypatch):
    _inject(monkeypatch)
    assert sc.get_s3_options() == {
        "key": "AKIAIOSFODNN7EXAMPLE",
        "secret": "wJalrXUtnFEMI-K7MDENG-bPxRfiCYEXAMPLEKEY",
        "endpoint_url": ENDPOINT,
    }


def test_endpoint_name_used_by_the_manifests_is_accepted(monkeypatch):
    """Every manifest in k8s/ injects AWS_ENDPOINT_URL; S3_ENDPOINT_URL also works."""
    _inject(monkeypatch, "S3_ENDPOINT_URL")
    assert sc.get_s3_options()["endpoint_url"] == ENDPOINT


def test_s3_endpoint_url_wins_over_the_sdk_wide_variable(monkeypatch):
    _inject(monkeypatch, "AWS_ENDPOINT_URL")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://127.0.0.1:8080")
    assert sc.get_s3_options()["endpoint_url"] == "http://127.0.0.1:8080"


# ------------------------------------------------------------- failures ----
@pytest.mark.parametrize("missing", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
def test_missing_credential_raises_environment_error(monkeypatch, missing):
    _inject(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(EnvironmentError) as exc:
        sc.get_s3_options()
    assert missing in str(exc.value)


def test_missing_endpoint_raises_environment_error(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI-K7MDENG-bPxRfiCYEXAMPLEKEY")
    with pytest.raises(EnvironmentError) as exc:
        sc.get_s3_options()
    assert "S3_ENDPOINT_URL" in str(exc.value)


def test_empty_string_is_not_a_credential(monkeypatch):
    """An env var set to "" must fail loudly instead of reaching s3fs with empty keys."""
    _inject(monkeypatch)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "")
    with pytest.raises(EnvironmentError):
        sc.get_s3_options()


# --------------------------------------------------- no kubectl anywhere ----
def _code_tokens(source: str) -> set:
    """String literals, names and attributes *in code* - docstrings and comments removed.

    The prose in these modules explains why kubectl is gone; the guard must judge the
    executable code, so docstrings are stripped before matching.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)
    tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.add(node.value)
        elif isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
    return tokens


@pytest.mark.parametrize("module", [sc, tp, utd])
def test_no_kubectl_or_subprocess_in_the_credential_path(module):
    source = (ROOT / "src" / f"{module.__name__}.py").read_text(encoding="utf-8")
    tokens = _code_tokens(source)
    assert "kubectl" not in tokens
    assert "subprocess" not in tokens
    assert "b64decode" not in tokens
    assert not any(token.startswith("rook-ceph-object-user") for token in tokens)
    assert not hasattr(module, "subprocess")


def test_pipelines_share_one_helper(monkeypatch):
    """hw_training_pipeline / hw_forward_training_pipeline import it from training_pipeline."""
    _inject(monkeypatch)
    assert tp.get_s3_options is sc.get_s3_options
    assert utd.get_s3_options is sc.get_s3_options
    assert tp.get_s3_options()["endpoint_url"] == ENDPOINT


def test_manifest_injects_the_variables_the_code_reads():
    """k8s/training-cronjob.yaml must keep providing exactly what get_s3_options() needs."""
    manifest = (ROOT / "k8s" / "training-cronjob.yaml").read_text(encoding="utf-8")
    assert "name: AWS_ACCESS_KEY_ID" in manifest
    assert "name: AWS_SECRET_ACCESS_KEY" in manifest
    assert "name: AWS_ENDPOINT_URL" in manifest
    assert "secretKeyRef" in manifest and "name: s3-credentials" in manifest
