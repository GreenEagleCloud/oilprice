"""S3/Ceph object-store options read straight from the environment.

Every workload that touches the Rook-Ceph RGW (training pipelines, feature
pipelines, the raw-data uploader) gets its credentials the same way: the k8s
manifests inject them as env vars from the secret "s3-credentials"
(secretKeyRef keys aws_access_key_id / aws_secret_access_key) plus a plain
endpoint env var pointing at the in-cluster RGW service.

Why not `kubectl get secret rook-ceph-object-user-...`?
    * It needs a kubeconfig and an RBAC role that may read Secrets in the
      *rook-ceph* namespace - a cross-namespace secret read from inside a Job
      pod is exactly the permission you do not want a training job to hold.
    * It pipes base64 credentials through an extra process and a second
      namespace's API surface for no benefit.
    * The image does not even ship kubectl, so the call only ever "worked"
      where a developer ran it from a terminal.

s3fs does NOT read FSSPEC_S3_* environment variables; S3FileSystem only accepts
key/secret/endpoint_url, so the values are handed over explicitly:
https://s3fs.readthedocs.io/en/latest/index.html#s3-compatible-storage
"""

import os

try:  # optional convenience for local runs (same .env the Hopsworks pipelines read)
    from pathlib import Path

    from dotenv import load_dotenv
except ImportError:  # the credential path itself must stay dependency-free
    pass
else:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

S3_ACCESS_KEY_ENV = "AWS_ACCESS_KEY_ID"
S3_SECRET_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
# AWS_ENDPOINT_URL is the botocore standard used by every manifest in k8s/ and
# by boto3 clients elsewhere in this repo; S3_ENDPOINT_URL is accepted as an
# explicit alias so a local run can name the endpoint without overriding the
# SDK-wide variable.
S3_ENDPOINT_ENVS = ("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL")


def get_s3_options() -> dict:
    """Read S3/Ceph object store options directly from secure environment variables.

    Eliminates kubectl execution vulnerabilities completely.
    """
    access_key = os.environ.get(S3_ACCESS_KEY_ENV)
    secret_key = os.environ.get(S3_SECRET_KEY_ENV)
    endpoint = next(
        (os.environ[name] for name in S3_ENDPOINT_ENVS if os.environ.get(name)),
        None,
    )

    if not all([access_key, secret_key, endpoint]):
        missing = [name for name in (S3_ACCESS_KEY_ENV, S3_SECRET_KEY_ENV) if not os.environ.get(name)]
        if not endpoint:
            missing.append(f"{S3_ENDPOINT_ENVS[0]} (or {S3_ENDPOINT_ENVS[1]})")
        raise EnvironmentError(
            "Missing critical S3 storage credentials in environment variables: "
            f"{', '.join(missing)}. "
            "In Kubernetes they come from secretKeyRef 's3-credentials' "
            "(aws_access_key_id / aws_secret_access_key) plus the RGW endpoint env var; "
            "locally export them before running the pipeline."
        )

    print(f"S3 Infrastructure Endpoint Connected: {endpoint}")
    return {"key": access_key, "secret": secret_key, "endpoint_url": endpoint}
