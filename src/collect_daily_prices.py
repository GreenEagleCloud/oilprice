import re
import sys
import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from io import BytesIO

BUCKET = "oilprice-features"
KEY = "raw/oil_data.csv"
# the old monthly LIK index file (dates like 2025-01) is kept under this key
# whenever it is replaced by the daily heizoel24 history
BACKUP_KEY = "raw/oil_data_lik_monthly.csv"

# heizoel24.ch chart API (unofficial, reverse-engineered from the site's JS bundles):
#   countryId=1  -> Switzerland (CH)
#   productId=1  -> heating oil (Heizöl)
#   rangeType=1  -> last 3 months (daily collection window, covers late-published days)
#   rangeType=11 -> full history since 2003 (bootstrap / re-sync)
API_URL = "https://www.heizoel24.ch/api/site/1/prices/history"
HEADERS = {"User-Agent": "Mozilla/5.0"}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def fetch_prices(range_type):
    """Fetch price rows from the heizoel24.ch chart API as a DataFrame [date, price, demand]."""
    response = requests.get(
        API_URL,
        params={"amount": 3000, "productId": 1, "rangeType": range_type},
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


def load_data_from_s3(bucket, key):
    """Load the training data file from S3. Returns None if the file does not exist yet."""
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "NotFound", "404"):
            return None
        raise
    return pd.read_csv(BytesIO(response["Body"].read()))


def save_data_to_s3(data, bucket, key):
    s3 = boto3.client("s3")
    csv_buffer = BytesIO()
    data.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())


def s3_object_exists(bucket, key):
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def backup_s3_object(bucket, src_key, dst_key):
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=src_key)["Body"].read()
    s3.put_object(Bucket=bucket, Key=dst_key, Body=body)


def merge(existing, new):
    """Merge newly fetched rows into the existing dataset; on overlapping dates the new value wins."""
    if "demand" not in existing.columns:
        existing = existing.copy()
        existing["demand"] = float("nan")
    merged = pd.concat([existing, new], ignore_index=True)
    return merged.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)


def main():
    existing = load_data_from_s3(BUCKET, KEY)
    legacy = (
        existing[~existing["date"].astype(str).str.match(DATE_RE)]
        if existing is not None
        else pd.DataFrame()
    )

    if existing is None or "--bootstrap" in sys.argv or len(legacy) > 0:
        # cold start, forced re-sync, or a legacy (e.g. monthly LIK) file found:
        # fetch the full daily history since 2003
        new = fetch_prices(11)
        print(f"fetched full history since 2003: {len(new)} rows ({new['date'].iloc[0]} .. {new['date'].iloc[-1]})")
        if existing is not None:
            print(f"existing file: {len(existing)} rows")
            if len(legacy) > 0:
                if not s3_object_exists(BUCKET, BACKUP_KEY):
                    backup_s3_object(BUCKET, KEY, BACKUP_KEY)
                    print(f"backed up legacy file ({len(legacy)} non-daily rows) to s3://{BUCKET}/{BACKUP_KEY}")
                else:
                    print(f"backup already exists at s3://{BUCKET}/{BACKUP_KEY}, replacing without new backup")
                merged = new.reset_index(drop=True)
            else:
                merged = merge(existing, new)
        else:
            print(f"no existing file at s3://{BUCKET}/{KEY}, starting fresh")
            merged = new.reset_index(drop=True)
    else:
        # daily collection: fetch the last 3 months and append new records
        new = fetch_prices(1)
        print(f"fetched last 3 months: {len(new)} rows ({new['date'].iloc[0]} .. {new['date'].iloc[-1]})")
        print(f"existing training data: {len(existing)} rows ({existing['date'].iloc[0]} .. {existing['date'].iloc[-1]})")
        merged = merge(existing, new)

    added = len(merged) - (0 if existing is None else len(existing))
    save_data_to_s3(merged, BUCKET, KEY)
    print(f"saved s3://{BUCKET}/{KEY}: {len(merged)} rows ({added} new records)")


if __name__ == "__main__":
    main()
