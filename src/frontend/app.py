import json
import math
import os
import time
from io import BytesIO

import boto3
import numpy as np
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


# ---------------------------------------------------------------------------
# Rhine data: gauge snapshot + daily gauge series + monthly port cargo
# (written by src/collect_rhein_levels.py and src/collect_rhein_cargo.py).
# S3 first, bundled CSV fallback - same pattern as the price history above.
# ---------------------------------------------------------------------------
RHEIN_S3_BUCKET = os.environ.get("RHEIN_S3_BUCKET", "oilprice-features")
RHEIN_SNAPSHOT_KEY = "raw/rhein_snapshot.json"
RHEIN_LEVELS_KEY = "raw/rhein_levels.csv"
RHEIN_CARGO_KEY = "raw/rhein_cargo.csv"

# Low-water thresholds. Mirrors src/rhein_config.py - the frontend image only
# contains src/frontend (see Dockerfile.frontend), so the constants cannot be
# imported from the collectors there.
BASEL_LOW_WATER_CM = float(os.environ.get("BASEL_LOW_WATER_CM", 350.0))
KAUB_LOW_WATER_CM = float(os.environ.get("KAUB_LOW_WATER_CM", 30.0))

RHEIN_LEVEL_DAYS = int(os.environ.get("RHEIN_LEVEL_DAYS", 365))      # gauge chart window
RHEIN_CARGO_MONTHS = int(os.environ.get("RHEIN_CARGO_MONTHS", 36))  # cargo chart window
RHEIN_CACHE_TTL = int(os.environ.get("RHEIN_CACHE_TTL", 15 * 60))

# Cargo groups shown in the chart (NST-2007 codes; the legacy classification
# uses group 3 for mineral oil - see src/collect_rhein_cargo.py).
_OIL_GROUPS = ("07", "3")
_HEATING_OIL_GROUP = "07.23"
_TOTAL_GROUP = "TOTAL"

_rhein_cache = {"ts": 0.0, "data": None}


def _rhein_local_candidates(key: str) -> list:
    """Bundled (docker: /app/data/rhein/...) or repo-local fallback paths."""
    base = os.path.dirname(os.path.abspath(__file__))
    rel = key[len("raw/"):] if key.startswith("raw/") else key
    return [
        os.path.join(base, "data", "rhein", rel),  # docker image: /app/data/...
        os.path.abspath(os.path.join(base, "..", "..", "data", "rhein", rel)),  # local dev
    ]


def _read_rhein_object(key: str) -> tuple:
    """Read one Rhine file. Returns (payload, source) with source 's3' or 'local'."""
    try:
        s3 = boto3.client("s3")
        return s3.get_object(Bucket=RHEIN_S3_BUCKET, Key=key)["Body"].read(), "s3"
    except Exception as e:
        candidates = _rhein_local_candidates(key)
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            raise RuntimeError(f"s3://{RHEIN_S3_BUCKET}/{key} unavailable ({e}); no bundled file")
        print(f"warning: s3 read of {key} failed ({e}); using {path}")
        with open(path, "rb") as fh:
            return fh.read(), "local"


def _numbers(series) -> list:
    """JSON-safe list: NaN/None -> None, everything else float."""
    return [
        None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        for v in series
    ]


def _read_rhein_snapshot() -> tuple:
    """Current gauge snapshot (Basel + Kaub + bottleneck), realtime source."""
    body, source = _read_rhein_object(RHEIN_SNAPSHOT_KEY)
    return json.loads(body.decode("utf-8")), source


def _read_rhein_levels() -> tuple:
    """Daily gauge series for the chart (last RHEIN_LEVEL_DAYS days).

    Mirrors src/hw_rhein_feature_pipeline.py: the daily MINIMUM is the
    shipping-relevant value, and the bottleneck is min(Basel, Kaub) with a
    fallback to Basel while Kaub has no readings (WSV publishes only ~30 days).
    """
    body, source = _read_rhein_object(RHEIN_LEVELS_KEY)
    df = pd.read_csv(BytesIO(body))
    df["date"] = pd.to_datetime(df["date"])
    daily = (
        df.sort_values("date")
        .pivot_table(index="date", columns="station", values="level_cm_min", aggfunc="last")
        .sort_index()
    )
    for station in ("basel", "kaub"):
        if station not in daily.columns:
            daily[station] = np.nan

    daily["basel"] = daily["basel"].ffill(limit=2)  # a single missing day stays usable
    daily["kaub"] = daily["kaub"].ffill(limit=2)
    daily["bottleneck"] = daily[["basel", "kaub"]].min(axis=1)
    low = (daily["basel"] < BASEL_LOW_WATER_CM) | (daily["kaub"] < KAUB_LOW_WATER_CM)
    daily["low_water_flag"] = low.fillna(False).astype(int)

    daily = daily.tail(max(1, RHEIN_LEVEL_DAYS))
    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in daily.index],
        "basel": _numbers(daily["basel"]),
        "kaub": _numbers(daily["kaub"]),
        "bottleneck": _numbers(daily["bottleneck"]),
        "low_water_flag": [int(v) for v in daily["low_water_flag"]],
    }
    return series, source


def _read_rhein_cargo() -> tuple:
    """Monthly mineral-oil cargo of the Port of Switzerland (lagged data)."""
    body, source = _read_rhein_object(RHEIN_CARGO_KEY)
    df = pd.read_csv(BytesIO(body))
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["month"] + "-01"), freq="M")

    def series_for(groups, direction):
        selected = df[df["group"].isin(groups) & (df["direction"] == direction)]
        if selected.empty:
            return pd.Series(dtype="float64")
        # min(): duplicated rows of overlapping bulletins must agree
        return selected.groupby("month")["tonnes"].min()

    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    frame = pd.DataFrame(
        {
            "oil_zufuhr": series_for(_OIL_GROUPS, "zufuhr").reindex(months),
            "oil_abfuhr": series_for(_OIL_GROUPS, "abfuhr").reindex(months),
            "heating_oil_zufuhr": series_for((_HEATING_OIL_GROUP,), "zufuhr").reindex(months),
            "total_zufuhr": series_for((_TOTAL_GROUP,), "zufuhr").reindex(months),
        }
    )
    published = df.sort_values("published_date").groupby("month")["published_date"].last()
    frame["published_date"] = published.reindex(months)

    frame = frame.tail(max(1, RHEIN_CARGO_MONTHS))
    known = frame[frame["oil_zufuhr"].notna()]
    cargo = {
        "months": [p.strftime("%Y-%m") for p in frame.index],
        "oil_zufuhr": _numbers(frame["oil_zufuhr"]),
        "oil_abfuhr": _numbers(frame["oil_abfuhr"]),
        "heating_oil_zufuhr": _numbers(frame["heating_oil_zufuhr"]),
        "total_zufuhr": _numbers(frame["total_zufuhr"]),
        "latest_month": known.index[-1].strftime("%Y-%m") if len(known) else None,
        "latest_published": str(known["published_date"].iloc[-1]) if len(known) else None,
    }
    return cargo, source


def _read_rhein() -> dict:
    """Snapshot + gauge history + cargo history, with per-section error notes."""
    payload = {"thresholds": {"basel_cm": BASEL_LOW_WATER_CM, "kaub_cm": KAUB_LOW_WATER_CM}}
    errors = {}
    sources = []
    for name, reader in (
        ("snapshot", _read_rhein_snapshot),
        ("levels", _read_rhein_levels),
        ("cargo", _read_rhein_cargo),
    ):
        try:
            data, source = reader()
            payload[name] = data
            sources.append(source)
        except Exception as e:
            print(f"warning: rhein {name} unavailable: {e}")
            payload[name] = None
            errors[name] = str(e)
    payload["source"] = "s3" if sources and all(s == "s3" for s in sources) else "local"
    payload["errors"] = errors
    return payload


@app.route('/api/rhein')
def api_rhein():
    """Rhine gauge snapshot + gauge history + port cargo history (cached).

    Gauges are realtime; the cargo figures are published 4-7 weeks after the
    reporting month, so they are displayed as history only and are never used
    as an input to a live prediction.
    """
    now = time.time()
    if _rhein_cache["data"] is None or now - _rhein_cache["ts"] > RHEIN_CACHE_TTL:
        _rhein_cache["ts"] = now
        _rhein_cache["data"] = _read_rhein()
    return jsonify(_rhein_cache["data"])


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