import os
import re
import sys
import time
import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from datetime import date
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

# The chart endpoint is unofficial, so every answer it gives is treated as
# untrusted input. The 2026-10-25..27 outage was a fast, repeated failure of
# exactly this call (7 job attempts in 5m43s, seven S3 reads and no write), and
# the collector had no retry at all - unlike the market collector, whose Yahoo /
# FRED fetches have retried with backoff since the beginning.
FETCH_ATTEMPTS = int(os.environ.get("PRICE_FETCH_ATTEMPTS", 5))
FETCH_TIMEOUT = int(os.environ.get("PRICE_FETCH_TIMEOUT", 60))
FETCH_MAX_BACKOFF = int(os.environ.get("PRICE_FETCH_MAX_BACKOFF", 60))
# "DateTime" and "Price" are what the training file needs; "Demand" is optional
# so a feed that stops serving the demand index does not break the price run.
REQUIRED_COLUMNS = ("DateTime", "Price")
# heizoel24 publishes the current day late, so a feed lagging further than this
# behind the clock is stale and worth investigating.
STALE_WARN_DAYS = int(os.environ.get("PRICE_STALE_WARN_DAYS", 3))


class PriceApiError(RuntimeError):
    """The heizoel24 answer was not the chart payload we expect."""


def _rows_to_frame(payload):
    """Chart payload -> DataFrame [date, price, demand], or PriceApiError.

    Guards against the shapes that used to crash the job with a bare KeyError
    (empty list, an error object instead of a list, rows without the expected
    keys) and against rows that would poison the training file.
    """
    if not isinstance(payload, list) or not payload:
        raise PriceApiError(
            f"unexpected payload: {type(payload).__name__} with {len(payload)} rows"
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in payload[0]]
    if missing:
        raise PriceApiError(
            f"payload rows lack {missing}; first row keys: {sorted(payload[0])}"
        )

    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(df["DateTime"], errors="coerce").dt.date.astype(str)
    df["price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["demand"] = (
        pd.to_numeric(df["Demand"], errors="coerce")
        if "Demand" in df.columns
        else float("nan")
    )
    out = df[["date", "price", "demand"]]

    unusable = out["price"].isna() | ~out["date"].str.match(DATE_RE)
    if unusable.any():
        print(f"  dropped {int(unusable.sum())} unusable rows (no numeric price or no ISO date)")
        out = out[~unusable]
    if out.empty:
        raise PriceApiError("payload had no usable rows")

    return (
        out.drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_prices(range_type, attempts=FETCH_ATTEMPTS, timeout=FETCH_TIMEOUT, sleep=None):
    """Fetch price rows from the heizoel24.ch chart API as a DataFrame [date, price, demand].

    Transient upstream problems (HTTP 5xx, timeouts, connection resets, an empty
    or malformed payload) are retried with exponential backoff, capped at
    FETCH_MAX_BACKOFF, so one bad minute at 23:30 UTC does not cost a day of
    history. `sleep` is injectable for tests; it is resolved at call time so a
    patched `time.sleep` is honoured.
    """
    if sleep is None:
        sleep = time.sleep
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                API_URL,
                params={"amount": 3000, "productId": 1, "rangeType": range_type},
                headers=HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            frame = _rows_to_frame(response.json())
            if attempt:
                print(f"  heizoel24 attempt {attempt + 1}/{attempts} succeeded")
            return frame
        except (requests.RequestException, PriceApiError, ValueError) as exc:
            last_error = exc
            print(f"  heizoel24 attempt {attempt + 1}/{attempts} failed: {exc}")
        if attempt < attempts - 1:
            wait = min(2 ** attempt, FETCH_MAX_BACKOFF)
            print(f"  retrying in {wait}s")
            sleep(wait)
    raise RuntimeError(
        f"heizoel24 API unavailable after {attempts} attempts (last error: {last_error})"
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


def warn_if_stale(frame, today=None, max_age_days=STALE_WARN_DAYS):
    """Warn (never fail) when the newest fetched price lags behind the clock.

    heizoel24 publishes the current day late, so a small lag is normal. A large
    one means the feed stopped serving new rows while still answering 200 - the
    failure mode that left the frontend showing prices weeks out of date.
    """
    today = today or date.today()
    newest = date.fromisoformat(frame["date"].iloc[-1])
    age = (today - newest).days
    if age > max_age_days:
        print(
            f"WARNING: newest fetched price is {newest} ({age} days old, threshold "
            f"{max_age_days}d) - the heizoel24 feed looks stale"
        )
    return age


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
        warn_if_stale(new)
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
        warn_if_stale(new)
        print(f"existing training data: {len(existing)} rows ({existing['date'].iloc[0]} .. {existing['date'].iloc[-1]})")
        merged = merge(existing, new)

    added = len(merged) - (0 if existing is None else len(existing))
    if added <= 0:
        # The feed answered but brought nothing new: the training file would age
        # silently, which is exactly how the frontend ended up showing old prices.
        print(f"WARNING: no new price records in this run (newest fetched date {new['date'].iloc[-1]})")
    save_data_to_s3(merged, BUCKET, KEY)
    print(f"saved s3://{BUCKET}/{KEY}: {len(merged)} rows ({added} new records), newest {merged['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
