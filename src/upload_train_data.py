"""Manage the raw training data in S3: s3://traindata/train_oilprice_raw.parquet.

Modes:
  - default (bootstrap): upload the full local history
    (data/heizoel24_heizoel_daily_ch_full.csv) as a single Parquet file. Run once.
  - --append: fetch recent daily prices from the heizoel24 chart API and append
    the new dates (in practice: the single record of the day) so the training
    data stays up to date. No daily job is needed - run this manually.

Credentials: s3fs does NOT read FSSPEC_S3_* environment variables; S3FileSystem
only accepts key/secret/endpoint_url. Therefore the Rook-generated secret is
read via kubectl and the options are passed explicitly:
https://s3fs.readthedocs.io/en/latest/index.html#s3-compatible-storage

Usage:
  python src/upload_train_data.py            # one-time bootstrap upload
  python src/upload_train_data.py --append   # append the record(s) of the day
"""

import argparse
import base64
import json
import subprocess
from pathlib import Path

import pandas as pd
import requests
import s3fs

SECRET_NAME = "rook-ceph-object-user-objectstore-objectstore-user"
BUCKET = "traindata"
KEY = "train_oilprice_raw.parquet"
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "heizoel24_heizoel_daily_ch_full.csv"

# heizoel24.ch chart API (same source as src/collect_daily_prices.py):
#   productId=1 -> heating oil (Heizoeil), rangeType=1 -> last 3 months
#   (fetching 3 months makes --append a safe catch-up if it is run late)
API_URL = "https://www.heizoel24.ch/api/site/1/prices/history"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_s3_options():
    """Read the S3 credentials from the Rook-generated secret in the rook-ceph namespace."""
    out = subprocess.run(
        [
            "kubectl",
            "-n",
            "rook-ceph",
            "get",
            "secret",
            SECRET_NAME,
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    secret_data = json.loads(out)["data"]

    # the values in the secret are base64-encoded
    access_key = base64.b64decode(secret_data["AccessKey"]).decode()
    secret_key = base64.b64decode(secret_data["SecretKey"]).decode()
    endpoint = base64.b64decode(secret_data["Endpoint"]).decode()
    return {
        "key": access_key,
        "secret": secret_key,
        "endpoint_url": endpoint,
    }


def fetch_recent_prices():
    """Fetch recent daily prices from the heizoel24 chart API as [date, price, demand]."""
    response = requests.get(
        API_URL,
        params={"amount": 3000, "productId": 1, "rangeType": 1},
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date.astype(str)
    return (
        df[["date", "Price", "Demand"]]
        .rename(columns={"Price": "price", "Demand": "demand"})
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def bootstrap(s3_options, s3):
    """Upload the full local history as Parquet (overwrites any existing file)."""
    df = pd.read_csv(DATA_FILE)
    df.to_parquet(f"s3://{BUCKET}/{KEY}", storage_options=s3_options)
    print(
        f"uploaded s3://{BUCKET}/{KEY}: {len(df)} rows "
        f"({df['date'].iloc[0]} .. {df['date'].iloc[-1]})"
    )


def append(s3_options, s3):
    """Append new daily records (typically the single record of the day)."""
    if not s3.exists(f"{BUCKET}/{KEY}"):
        raise SystemExit(
            f"s3://{BUCKET}/{KEY} does not exist yet - run the bootstrap first: "
            "python src/upload_train_data.py"
        )
    existing = pd.read_parquet(f"s3://{BUCKET}/{KEY}", storage_options=s3_options)
    new = fetch_recent_prices()
    merged = (
        pd.concat([existing, new], ignore_index=True)
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged.to_parquet(f"s3://{BUCKET}/{KEY}", storage_options=s3_options)
    added = len(merged) - len(existing)
    print(
        f"updated s3://{BUCKET}/{KEY}: {len(merged)} rows "
        f"({added} new record(s), latest: {merged['date'].iloc[-1]})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--append",
        action="store_true",
        help="append the latest daily record(s) from the heizoel24 API "
        "instead of bootstrapping from the local CSV",
    )
    args = parser.parse_args()

    s3_options = get_s3_options()
    s3 = s3fs.S3FileSystem(**s3_options)
    if not s3.exists(BUCKET):
        s3.mkdir(BUCKET)

    if args.append:
        append(s3_options, s3)
    else:
        bootstrap(s3_options, s3)


if __name__ == "__main__":
    main()
