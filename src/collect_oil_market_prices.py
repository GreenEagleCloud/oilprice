"""Collect the market drivers of the Swiss heating-oil purchase price.

Two daily series are collected (see src/market_config.py for the sources):

  * ICE Gasoil close in USD per tonne  (product price of the refined product)
  * USD/CHF spot close                 (currency of the purchase)

and converted into the unit of the prediction target:

    gasoil_chf_tonne      = gasoil_usd_tonne * usd_chf
    market_price_chf_100l = gasoil_chf_tonne / 11.8343   (CHF per 100 l)

Output (one row per trading day):
    s3://oilprice-features/raw/oil_market_prices.csv
    (fallback when no S3 credentials are set: data/oil/oil_market_prices.csv)

Rows are upserted by date, so a daily run refreshes the recent window (late
corrections) and keeps the full history.

Run:
    python src/collect_oil_market_prices.py                 # full history
    python src/collect_oil_market_prices.py --period 2y     # recent window
    python src/collect_oil_market_prices.py --no-s3
"""

import argparse
import io
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_config import (  # noqa: E402
    BARREL_PER_TONNE,
    BRENT_SYMBOL,
    DEFAULT_PERIOD,
    ECB_DAILY_URL,
    ECB_HIST_URL,
    ECB_HIST_ZIP_URL,
    FRED_BRENT_SERIES,
    FRED_CSV_URL,
    FX_SYMBOL,
    GASOIL_SYMBOL,
    HEIZOEL_100L_UNITS_PER_TONNE,
    LOCAL_MARKET_CSV,
    MARKET_RAW_KEY,
    MARKET_RAW_COLUMNS,
    S3_BUCKET,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (oilprice market data collector)"}
# FRED's edge rejects custom/Mozilla-style user agents from datacenter IPs
# (verified 2026-10-18: "Mozilla/5.0 (oilprice ...)" and "oilprice-collector/1.0"
# time out, the default python-requests agent returns 200 in ~0.2s), so the
# FRED download deliberately sends no custom User-Agent at all.
FRED_HEADERS = {"Accept": "text/csv,*/*"}


# ---------------------------------------------------------------------------
# Storage helpers (same pattern as src/collect_rhein_levels.py)
# ---------------------------------------------------------------------------
def _s3_enabled():
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))


def _s3():
    import boto3

    return boto3.client("s3")  # endpoint from AWS_ENDPOINT_URL


def load_csv(key, local_path):
    if _s3_enabled():
        from botocore.exceptions import ClientError

        try:
            body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "NotFound", "404"):
                return None
            raise
        return pd.read_csv(io.StringIO(body.decode("utf-8")))
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return None


def save_csv(df, key, local_path):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    if _s3_enabled():
        _s3().put_object(
            Bucket=S3_BUCKET, Key=key, Body=buffer.getvalue().encode("utf-8"), ContentType="text/csv"
        )
        print(f"saved s3://{S3_BUCKET}/{key} ({len(df)} rows)")
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as fh:
            fh.write(buffer.getvalue())
        print(f"saved {local_path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def _close_column(frame):
    """Return the Close series of a yfinance frame (single or MultiIndex cols)."""
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")
    if isinstance(frame.columns, pd.MultiIndex):
        # group_by="ticker" style: (ticker, Open/High/Low/Close/Volume)
        for level in frame.columns.get_level_values(-1).unique():
            if str(level).lower() == "close":
                series = frame.xs(level, level=-1, axis=1)
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series.astype("float64")
        return pd.Series(dtype="float64")
    for name in ("Close", "close"):
        if name in frame.columns:
            return frame[name].astype("float64")
    return pd.Series(dtype="float64")


def fetch_yahoo_close(symbol, period, attempts=3, timeout=60):
    """Daily closes of one Yahoo symbol as a DataFrame(date, close).

    Yahoo answers with HTTP 429 (rate limit) from time to time and yfinance
    reports that as an *empty* frame instead of raising, so an empty result is
    retried with a generous backoff (Yahoo's window is per minute) before it is
    treated as "symbol has no data" and the caller falls back to ECB/FRED.

    Note: yfinance must run in the main thread - wrapping it in a worker thread
    (to enforce a timeout) makes it return an empty frame. A stuck request is
    bounded by the Kubernetes job deadline instead (activeDeadlineSeconds).
    """
    import yfinance as yf

    close = pd.Series(dtype="float64")
    for attempt in range(attempts):
        try:
            frame = yf.download(symbol, period=period, progress=False, auto_adjust=False)
            close = _close_column(frame).dropna()
            if not close.empty:
                break
            print(f"  yfinance {symbol}: empty result (attempt {attempt + 1}, rate limit?)")
        except Exception as exc:  # transient Yahoo/HTTP problems
            print(f"  yfinance {symbol} attempt {attempt + 1} failed: {exc}")
        time.sleep(20 * (attempt + 1))

    close = close.dropna()
    if close.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({"date": pd.to_datetime(close.index).tz_localize(None).normalize(),
                        "close": close.to_numpy(dtype="float64")})
    return out.sort_values("date").reset_index(drop=True)


def fetch_fred(series_id):
    """Daily FRED observations as a DataFrame(date, value) - no API key needed."""
    last_error = None
    for attempt in range(4):
        try:
            resp = requests.get(FRED_CSV_URL, params={"id": series_id}, headers=FRED_HEADERS, timeout=90)
            if resp.status_code == 200:
                raw = pd.read_csv(io.StringIO(resp.text))
                date_col, value_col = raw.columns[0], raw.columns[1]
                value = pd.to_numeric(raw[value_col], errors="coerce")  # FRED "." -> NaN
                out = pd.DataFrame({"date": pd.to_datetime(raw[date_col]).dt.normalize(),
                                    "value": value})
                return out.dropna().sort_values("date").reset_index(drop=True)
            last_error = RuntimeError(f"{series_id} -> HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"FRED {series_id} unavailable: {last_error}")


def fetch_gasoil_usd_tonne(period):
    """ICE Gasoil in USD/tonne, with a documented Brent proxy as fallback."""
    # The Yahoo gasoil symbol is opt-in (see src/market_config.py): Yahoo
    # delisted ICE Gasoil, and the guaranteed 404 makes Yahoo rate-limit the
    # follow-up requests, so it is only requested when configured explicitly.
    if GASOIL_SYMBOL:
        frame = fetch_yahoo_close(GASOIL_SYMBOL, period, attempts=2, timeout=45)
        if not frame.empty:
            frame["gasoil_usd_tonne"] = frame["close"]
            frame["gasoil_source"] = f"yahoo:{GASOIL_SYMBOL}"
            return frame[["date", "gasoil_usd_tonne", "gasoil_source"]]
        print(f"WARNING: Yahoo symbol {GASOIL_SYMBOL} returned no data; "
              "falling back to a Brent proxy")

    # Yahoo Brent is only tried twice; FRED (below) is the reliable fallback and
    # answers in well under a second.
    frame = fetch_yahoo_close(BRENT_SYMBOL, period, attempts=2, timeout=45)
    if not frame.empty:
        frame["gasoil_usd_tonne"] = frame["close"] * BARREL_PER_TONNE
        frame["gasoil_source"] = f"brent_proxy:yahoo:{BRENT_SYMBOL}"
        return frame[["date", "gasoil_usd_tonne", "gasoil_source"]]

    frame = fetch_fred(FRED_BRENT_SERIES)
    frame["gasoil_usd_tonne"] = frame["value"] * BARREL_PER_TONNE
    frame["gasoil_source"] = f"brent_proxy:fred:{FRED_BRENT_SERIES}"
    return frame[["date", "gasoil_usd_tonne", "gasoil_source"]]


def _parse_ecb_daily(text) -> pd.DataFrame:
    """Parse eurofxref-daily.xml into DataFrame(date, usd_chf).

    The ECB quotes are foreign units per 1 EUR, so USD/CHF = EURCHF / EURUSD.
    """
    rows = []
    for block in re.finditer(r"<Cube time='(\d{4}-\d{2}-\d{2})'>(.*?)</Cube>", text, re.S):
        day, body = block.group(1), block.group(2)
        usd = re.search(r"currency='USD' rate='([\d.]+)'", body)
        chf = re.search(r"currency='CHF' rate='([\d.]+)'", body)
        if usd and chf:
            rows.append((day, float(chf.group(1)) / float(usd.group(1))))
    return pd.DataFrame(rows, columns=["date", "usd_chf"])


def _usd_chf_from_hist(raw) -> pd.DataFrame:
    """Parse the ECB eurofxref history CSV (bytes or text) into (date, usd_chf)."""
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    table = pd.read_csv(io.StringIO(text))
    usd = pd.to_numeric(table["USD"], errors="coerce")
    chf = pd.to_numeric(table["CHF"], errors="coerce")
    return pd.DataFrame({"date": pd.to_datetime(table["Date"]).dt.normalize(),
                         "usd_chf": chf / usd})


def _fetch_ecb_hist() -> pd.DataFrame:
    """ECB euro reference rate history, preferring the zip archive.

    The plain eurofxref-hist.csv is delivered truncated by some CDNs, which cuts
    the FX leg (and with it every market driver) off in 2010 while still looking
    like a successful download. eurofxref-hist.zip holds the same CSV complete,
    so it is the primary source and the .csv is only the fallback.
    """
    try:
        resp = requests.get(ECB_HIST_ZIP_URL, headers=HEADERS, timeout=90)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
                name = next(n for n in archive.namelist() if n.endswith(".csv"))
                frame = _usd_chf_from_hist(archive.read(name))
            if not frame.empty:
                print(f"  ECB USD/CHF history (zip): {len(frame)} rows "
                      f"{frame['date'].min().date()} .. {frame['date'].max().date()}")
                return frame
        print(f"  ECB history zip -> HTTP {resp.status_code}")
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as exc:
        print(f"  ECB history zip unavailable: {exc}")

    try:
        resp = requests.get(ECB_HIST_URL, headers=HEADERS, timeout=90)
        if resp.status_code == 200:
            frame = _usd_chf_from_hist(resp.content)
            print(f"  ECB USD/CHF history (csv): {len(frame)} rows "
                  f"{frame['date'].min().date()} .. {frame['date'].max().date()}")
            return frame
        print(f"  ECB history -> HTTP {resp.status_code}")
    except requests.RequestException as exc:
        print(f"  ECB history unavailable: {exc}")

    return pd.DataFrame(columns=["date", "usd_chf"])


def fetch_ecb_usd_chf() -> pd.DataFrame:
    """USD/CHF from the ECB euro reference rates (daily history since 1999)."""
    frames = [_fetch_ecb_hist()]

    # The history file sometimes lags one day behind the daily publication.
    try:
        resp = requests.get(ECB_DAILY_URL, headers=HEADERS, timeout=60)
        if resp.status_code == 200:
            daily = _parse_ecb_daily(resp.text)
            if not daily.empty:
                frames.append(daily)
    except requests.RequestException as exc:
        print(f"  ECB daily XML unavailable: {exc}")

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "usd_chf", "fx_source"])

    out = pd.concat(frames, ignore_index=True).dropna()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    # Plausibility guard: USD/CHF was 1.02..1.68 between 1999 and 2009 and
    # 0.82..1.20 since 2010. Anything outside 0.75..1.8 is a broken feed, not a
    # market move - keeping such rows would poison the CHF conversion of every
    # price day.
    plausible = out["usd_chf"].between(0.75, 1.8)
    if not plausible.all():
        print(f"  ECB USD/CHF: dropped {int((~plausible).sum())} implausible rows "
              f"(outside 0.75..1.8)")
    out = out.loc[plausible]

    out = out.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    out["fx_source"] = "ecb:eurofxref"
    print(f"  ECB USD/CHF: {len(out)} rows {out['date'].min().date()} .. {out['date'].max().date()}")
    return out[["date", "usd_chf", "fx_source"]]


def fetch_usd_chf(period):
    """USD/CHF spot close, with the ECB euro reference rates as fallback."""
    frame = fetch_yahoo_close(FX_SYMBOL, period)
    if not frame.empty:
        frame["usd_chf"] = frame["close"]
        frame["fx_source"] = f"yahoo:{FX_SYMBOL}"
        return frame[["date", "usd_chf", "fx_source"]]

    print(f"WARNING: Yahoo symbol {FX_SYMBOL} returned no data; falling back to ECB")
    return fetch_ecb_usd_chf()



# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------
def build_market_frame(period=DEFAULT_PERIOD) -> pd.DataFrame:
    """Merge the two drivers on the trading-day index and convert to CHF/100l."""
    # FX first: Yahoo serves the FX request reliably, and a failing product
    # lookup must not cost us the whole currency series.
    fx = fetch_usd_chf(period)
    gasoil = fetch_gasoil_usd_tonne(period)

    df = gasoil.merge(fx, on="date", how="outer").sort_values("date").reset_index(drop=True)

    # FX and ICE trade on different calendars (holidays). The FX leg is filled
    # in both directions (a rate always exists); the product price is only
    # filled forward, so a stale quote is never invented backwards. The limit
    # bounds the fill to a normal holiday/weekend gap - a source that stops
    # publishing must produce NULLs, not a value carried for months.
    df["usd_chf"] = df["usd_chf"].ffill(limit=3).bfill(limit=3)
    df["gasoil_usd_tonne"] = df["gasoil_usd_tonne"].ffill(limit=5)
    df["fx_source"] = df["fx_source"].ffill(limit=3).bfill(limit=3)
    df["gasoil_source"] = df["gasoil_source"].ffill(limit=5)

    df["gasoil_chf_tonne"] = (df["gasoil_usd_tonne"] * df["usd_chf"]).round(4)
    df["market_price_chf_100l"] = (df["gasoil_chf_tonne"] / HEIZOEL_100L_UNITS_PER_TONNE).round(4)

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["gasoil_usd_tonne", "usd_chf"]).reset_index(drop=True)
    df = df[MARKET_RAW_COLUMNS]

    print(
        f"market data: {len(df)} rows {df['date'].min()} .. {df['date'].max()} | "
        f"gasoil source={df['gasoil_source'].iloc[-1]} fx source={df['fx_source'].iloc[-1]}"
    )
    return df


def merge_raw(existing, new) -> pd.DataFrame:
    """Upsert the fresh window into the stored history by date."""
    if existing is None or existing.empty:
        return new
    existing = existing.copy()
    existing["date"] = existing["date"].astype(str)
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    return merged.sort_values("date").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ICE Gasoil + USD/CHF market data")
    parser.add_argument("--period", default=DEFAULT_PERIOD,
                        help="yfinance period (max, 10y, 5y, 2y, ...); default: max")
    parser.add_argument("--no-s3", action="store_true", help="force local files")
    args = parser.parse_args()

    if args.no_s3:
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    frame = build_market_frame(args.period)
    if frame.empty:
        print("ERROR: no market rows fetched", file=sys.stderr)
        return 1

    merged = merge_raw(load_csv(MARKET_RAW_KEY, LOCAL_MARKET_CSV), frame)
    save_csv(merged, MARKET_RAW_KEY, LOCAL_MARKET_CSV)
    print(f"collected_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

