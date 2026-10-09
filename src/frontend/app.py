import math
import os
import time
from io import BytesIO

import boto3
import pandas as pd
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend service URL (overridable via the BACKEND_URL env var, set in the
# k8s frontend deployment). Defaults to the in-cluster inference service.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://inference-service:5000/predict")

# Raw price data written daily by src/collect_daily_prices.py (source of truth).
PRICE_S3_URI = os.environ.get("PRICE_S3_URI", "s3://oilprice-features/raw/oil_data.csv")
PRICE_LOCAL_FILE = os.environ.get("PRICE_LOCAL_FILE", "")  # empty = auto-detect
_PRICE_FILENAME = "heizoel24_heizoel_daily_ch_full.csv"


def _local_price_candidates() -> list:
    base = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(base, "data", _PRICE_FILENAME),  # docker image: /app/data/...
        os.path.abspath(os.path.join(base, "..", "..", "data", _PRICE_FILENAME)),  # local dev: repo root
    ]
# How long to cache the price history between S3/local reads (seconds).
PRICE_CACHE_TTL = int(os.environ.get("PRICE_CACHE_TTL", 15 * 60))

_price_cache = {"ts": 0.0, "data": None}


def _read_price_history() -> tuple:
    """Load the heizoel24 daily price/demand series.

    Prefers the CSV written daily to S3 by the collect CronJob (fresh
    source of truth); falls back to the CSV bundled into the image
    (see Dockerfile.frontend) when S3 is not configured or reachable.

    Returns (series_dict, source) where source is 's3' or 'local'.
    """
    bucket, key = PRICE_S3_URI[len("s3://"):].split("/", 1)
    try:
        s3 = boto3.client("s3")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_csv(BytesIO(body))
        source = "s3"
    except Exception as e:
        print(f"warning: S3 read failed ({e}); falling back to bundled CSV")
        candidates = [PRICE_LOCAL_FILE] if PRICE_LOCAL_FILE else _local_price_candidates()
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            raise RuntimeError(f"no local price CSV found (tried {candidates})")
        df = pd.read_csv(path)
        source = "local"

    # normalize column names (collect writes date/price/demand; the 5y file
    # uses price_chf_per_100l)
    rename = {}
    if "price" not in df.columns:
        for col in df.columns:
            if col.lower() in ("price", "price_chf_per_100l", "preis"):
                rename[col] = "price"
                break
    if "demand" not in df.columns:
        for col in df.columns:
            if col.lower().startswith("demand"):
                rename[col] = "demand"
                break
    df = df.rename(columns=rename)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    prices = df["price"].astype(float).where(df["price"].notna(), None)
    demands = (
        df["demand"].astype(float).where(df["demand"].notna(), None)
        if "demand" in df.columns
        else [None] * len(df)
    )
    return (
        {
            "dates": [str(d) for d in df["date"]],
            "price": [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in prices],
            "demand": [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in demands],
        },
        source,
    )


@app.route('/api/history')
def api_history():
    """Price & demand history for the chart (cached, S3-first)."""
    now = time.time()
    if _price_cache["data"] is None or now - _price_cache["ts"] > PRICE_CACHE_TTL:
        _price_cache["ts"] = now
        _price_cache["data"] = _read_price_history()
    data, source = _price_cache["data"]
    return jsonify({"source": source, **data})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Proxy to the inference service.

    Passes the backend response through (JSON + status code) so users see the
    backend's error messages (e.g. "no feature vector for date ...; latest
    available date: ...") instead of a raw HTTPError string.
    """
    data = request.json
    try:
        response = requests.post(BACKEND_URL, json=data, timeout=30)
    except requests.RequestException as e:
        return jsonify({'error': f'prediction service is unreachable: {e}'}), 502

    try:
        payload = response.json()
    except ValueError:
        return jsonify({'error': f'prediction service returned a non-JSON response (HTTP {response.status_code})'}), 502

    if response.status_code != 200:
        if isinstance(payload, dict):
            return jsonify(payload), response.status_code
        return jsonify({'error': str(payload)}), response.status_code

    return jsonify(payload)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)