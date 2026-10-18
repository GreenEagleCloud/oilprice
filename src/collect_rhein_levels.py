"""Collect Rhine water-level / discharge data for Basel-Rheinhalle and Kaub.

Sources (all verified live 2026-10-17, see src/rhein_config.py):
  * data.bs.ch dataset 100089  Basel-Rheinhalle, 5-minute pegelhoehe [cm] and
                               abfluss [m3/s], timestamps in UTC, history since
                               2020-06-22. Paging limit offset+limit <= 10000,
                               so the history is fetched in month windows.
  * WSV Pegelonline            Basel-Rheinhalle + Kaub, parameter W [cm]
                               (Kaub additionally Q [m3/s]); timestamps are
                               local time and are normalised to UTC here.
  * LINDAS / BAFU              latest official Swiss observation of station
                               2289 (Basel-Rheinhalle) - cross-check + snapshot.

Output (one row per UTC calendar day and station):
    s3://oilprice-features/raw/rhein_levels.csv
plus a realtime snapshot:
    s3://oilprice-features/raw/rhein_snapshot.json
When no S3 credentials are set the files are written to data/rhein/ instead
(same fallback pattern as src/collect_daily_prices.py).

Gauge values are corrected after publication, so existing rows are only
overwritten when the freshly fetched value differs - and the change is logged.

Run:
    python src/collect_rhein_levels.py --bootstrap      # full history since 2020
    python src/collect_rhein_levels.py                  # daily update (35 days)
    python src/collect_rhein_levels.py --since 2025-09-01 --no-s3
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rhein_config import (  # noqa: E402
    BASEL,
    BOOTSTRAP_START,
    BS_CH_CSV_URL,
    BS_CH_MAX_LIMIT,
    DAILY_LOOKBACK_DAYS,
    KAUB,
    LINDAS_OBSERVATION_URL,
    LOCAL_LEVELS_CSV,
    LOCAL_SNAPSHOT_JSON,
    RHEIN_LEVELS_KEY,
    RHEIN_SNAPSHOT_KEY,
    S3_BUCKET,
    STATIONS,
    WSV_BASE,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (oilprice rhine gauge collector)"}
LEVEL_COLUMNS = [
    "date",
    "station",
    "level_cm_last",
    "level_cm_mean",
    "level_cm_min",
    "level_cm_max",
    "discharge_m3s_mean",
    "n_level_obs",
    "sources",
]


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _get(url, params=None, headers=None, timeout=60, retries=3):
    """GET with exponential backoff; raises RuntimeError on persistent failure."""
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            last_error = RuntimeError(f"{url} -> HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2 ** attempt)
    raise last_error


# ---------------------------------------------------------------------------
# Storage helpers (S3 when credentials are set, local files otherwise)
# ---------------------------------------------------------------------------
def _s3_enabled():
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))


def _s3():
    import boto3

    return boto3.client("s3")  # endpoint from AWS_ENDPOINT_URL


def load_csv(key, local_path):
    if _s3_enabled():
        import boto3
        from botocore.exceptions import ClientError

        try:
            body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "NotFound", "404"):
                return None
            raise
        return pd.read_csv(io.StringIO(body.decode("utf-8")))
    if local_path.exists():
        return pd.read_csv(local_path)
    return None


def save_csv(df, key, local_path):
    csv = df.to_csv(index=False)
    if _s3_enabled():
        _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=csv.encode("utf-8"))
        print(f"saved s3://{S3_BUCKET}/{key} ({len(df)} rows)")
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(csv, encoding="utf-8")
        print(f"saved {local_path} ({len(df)} rows)")


def save_json(obj, key, local_path):
    payload = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    if _s3_enabled():
        _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=payload.encode("utf-8"))
        print(f"saved s3://{S3_BUCKET}/{key}")
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(payload, encoding="utf-8")
        print(f"saved {local_path}")


# ---------------------------------------------------------------------------
# data.bs.ch (Basel-Rheinhalle, 5-minute series, UTC)
# ---------------------------------------------------------------------------
def _month_windows(start, end):
    """Yield [start, next_month) windows covering [start, end)."""
    cur = start
    while cur < end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        yield cur, min(nxt, end)
        cur = nxt


def fetch_bs_ch_window(dataset, start, end):
    """Fetch one time window of the data.bs.ch CSV export (max 10000 rows)."""
    params = {
        "timezone": "UTC",
        "sort": "timestamp",
        # ODSQL only accepts quoted date literals; unquoted timestamps either
        # raise ODSQLSyntaxError or compare numerically against the wrong year.
        "where": f'timestamp>="{start.isoformat()}" AND timestamp<"{end.isoformat()}"',
        "limit": BS_CH_MAX_LIMIT,
    }
    text = _get(BS_CH_CSV_URL.format(dataset=dataset), params=params).text
    df = pd.read_csv(io.StringIO(text), sep=";")
    if df.empty:
        return df
    if len(df) >= BS_CH_MAX_LIMIT:
        print(f"WARNING: window {start}..{end} hit the {BS_CH_MAX_LIMIT} row limit; split it")
    df = df.rename(columns={"pegelhoehe": "level_cm", "abfluss": "discharge_m3s", "pegel": "level_m_asl"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df


def fetch_bs_ch_history(cfg, since, until):
    """Fetch the Basel history from data.bs.ch in month windows."""
    frames = []
    for start, end in _month_windows(since, until):
        frame = fetch_bs_ch_window(cfg["bs_ch_dataset"], start, end)
        if frame.empty:
            continue
        frame["station"] = cfg["station"]
        frame["source"] = "data.bs.ch"
        frames.append(frame)
        print(f"  data.bs.ch {cfg['station']} {start}..{end}: {len(frame)} rows")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# WSV Pegelonline (Basel + Kaub, local time -> UTC)
# ---------------------------------------------------------------------------
def fetch_wsv_series(uuid, parameter, start=None, kind="measurements"):
    """Fetch a WSV time series; returns an empty frame when it does not exist."""
    params = {}
    if start:
        params["start"] = start
    resp = _get(f"{WSV_BASE}/stations/{uuid}/{parameter}/{kind}.json", params=params)
    data = resp.json()
    if isinstance(data, dict):  # e.g. {"status": 404, "message": "Timeseries does not exist."}
        return pd.DataFrame()
    frame = pd.DataFrame(data)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame


def fetch_wsv_current(uuid, parameter="W"):
    """Latest WSV measurement of one station/parameter (or None)."""
    resp = _get(f"{WSV_BASE}/stations/{uuid}/{parameter}/currentmeasurement.json")
    data = resp.json()
    if not isinstance(data, dict) or "timestamp" not in data:
        return None
    return {
        "timestamp": pd.to_datetime(data["timestamp"], utc=True, format="ISO8601"),
        "value": float(data["value"]),
    }


# ---------------------------------------------------------------------------
# LINDAS / BAFU (latest official Swiss observation)
# ---------------------------------------------------------------------------
def fetch_lindas_latest(station):
    """Latest BAFU observation of a Swiss hydrology station (local time -> UTC).

    NOTE on units: the BAFU/LINDAS `waterLevel` dimension is the gauge height in
    METRES ABOVE SEA LEVEL (e.g. 244.84), while data.bs.ch `pegelhoehe` and the
    WSV `W` parameter are centimetres above the local gauge datum (e.g. 484).
    The two are only comparable through `level_m_asl` (data.bs.ch column `pegel`).
    """
    resp = _get(
        LINDAS_OBSERVATION_URL.format(station=station),
        headers={**HEADERS, "Accept": "application/ld+json"},
    )
    doc = resp.json()

    def _value(name):
        entry = doc.get(name)
        if isinstance(entry, dict):
            raw = entry.get("@value")
            return raw if raw is not None else entry.get("@id")
        return None

    ts = _value("https://environment.ld.admin.ch/foen/hydro/dimension/measurementTime")
    level = _value("https://environment.ld.admin.ch/foen/hydro/dimension/waterLevel")
    discharge = _value("https://environment.ld.admin.ch/foen/hydro/dimension/discharge")
    return {
        "timestamp": pd.to_datetime(ts, utc=True, format="ISO8601") if ts else None,
        "level_m_asl": float(level) if level is not None else None,
        "discharge_m3s": float(discharge) if discharge is not None else None,
        "danger_level": _value("https://environment.ld.admin.ch/foen/hydro/dimension/dangerLevel"),
    }



# ---------------------------------------------------------------------------
# Daily aggregation + merge
# ---------------------------------------------------------------------------
def aggregate_daily(raw):
    """Aggregate 5-/15-minute readings into one row per UTC day and station."""
    if raw.empty:
        return pd.DataFrame(columns=LEVEL_COLUMNS)
    raw = raw.sort_values("timestamp").copy()
    raw["date"] = raw["timestamp"].dt.date

    rows = []
    for (station, day), grp in raw.groupby(["station", "date"]):
        levels = grp["level_cm"].dropna()
        discharge = grp["discharge_m3s"].dropna() if "discharge_m3s" in grp.columns else pd.Series(dtype="float64")
        sources = "|".join(sorted({s for s in grp["source"].dropna().unique()}))
        rows.append(
            {
                "date": day.isoformat(),
                "station": station,
                "level_cm_last": float(levels.iloc[-1]) if len(levels) else None,
                "level_cm_mean": round(float(levels.mean()), 1) if len(levels) else None,
                "level_cm_min": float(levels.min()) if len(levels) else None,
                "level_cm_max": float(levels.max()) if len(levels) else None,
                "discharge_m3s_mean": round(float(discharge.mean()), 1) if len(discharge) else None,
                "n_level_obs": int(len(levels)),
                "sources": sources,
            }
        )
    return pd.DataFrame(rows)[LEVEL_COLUMNS]


def merge_levels(existing, new):
    """Merge daily rows keyed by (date, station); the fresh value wins.

    Gauge series are corrected after publication, so a changed value is logged
    instead of silently replacing history (audit trail for backfills).
    """
    if existing is None or existing.empty:
        return new

    existing = existing.copy()
    existing["date"] = existing["date"].astype(str)
    new = new.copy()
    new["date"] = new["date"].astype(str)

    lookup = existing.drop_duplicates(subset=["date", "station"]).set_index(["date", "station"])
    for _, row in new.iterrows():
        key = (row["date"], row["station"])
        if key not in lookup.index:
            continue
        previous = lookup.loc[key, "level_cm_min"]
        fresh = row["level_cm_min"]
        if previous is not None and fresh is not None and abs(float(previous) - float(fresh)) > 0.5:
            print(f"  corrected {key[0]} {key[1]}: level_cm_min {previous} -> {fresh}")

    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "station"], keep="last")
    return merged.sort_values(["date", "station"]).reset_index(drop=True)


STATIONS_BY_NAME = {cfg["station"]: cfg for cfg in STATIONS}


# ---------------------------------------------------------------------------
# Realtime snapshot (for inference / dashboards)
# ---------------------------------------------------------------------------
def build_snapshot():
    """Collect the current gauge values of all stations plus the bottleneck."""
    snapshot = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stations": {},
    }

    for cfg in STATIONS:
        entry = {
            "name": cfg["name"],
            "rhine_km": cfg["rhine_km"],
            "low_water_cm": cfg["low_water_cm"],
        }

        current = fetch_wsv_current(cfg["wsv_uuid"])
        if current:
            entry["wsv_level_cm"] = current["value"]
            entry["wsv_timestamp_utc"] = current["timestamp"].isoformat()

        if cfg["wsv_discharge"]:
            discharge = fetch_wsv_current(cfg["wsv_uuid"], "Q")
            if discharge:
                entry["wsv_discharge_m3s"] = discharge["value"]

        if cfg["bs_ch_dataset"]:
            latest = fetch_bs_ch_window(
                cfg["bs_ch_dataset"], date.today() - timedelta(days=2), date.today() + timedelta(days=1)
            )
            if not latest.empty:
                last = latest.sort_values("timestamp").iloc[-1]
                entry["bs_ch_level_cm"] = float(last["level_cm"])
                entry["bs_ch_discharge_m3s"] = float(last["discharge_m3s"])
                entry["bs_ch_timestamp_utc"] = last["timestamp"].isoformat()

        if cfg["bafu_station"]:
            lindas = fetch_lindas_latest(cfg["bafu_station"])
            if lindas and lindas["timestamp"] is not None:
                # metres above sea level - a different datum than level_cm
                entry["lindas_level_m_asl"] = lindas["level_m_asl"]
                entry["lindas_discharge_m3s"] = lindas["discharge_m3s"]
                entry["lindas_danger_level"] = lindas["danger_level"]
                entry["lindas_timestamp_utc"] = lindas["timestamp"].isoformat()

        snapshot["stations"][cfg["station"]] = entry

    levels = {
        name: values.get("wsv_level_cm", values.get("bs_ch_level_cm"))
        for name, values in snapshot["stations"].items()
    }
    known = {name: value for name, value in levels.items() if value is not None}
    if known:
        bottleneck = min(known, key=lambda name: known[name])
        snapshot["bottleneck_station"] = bottleneck
        snapshot["bottleneck_level_cm"] = known[bottleneck]
        snapshot["low_water_flag"] = int(known[bottleneck] < STATIONS_BY_NAME[bottleneck]["low_water_cm"])
    return snapshot


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect Basel + Kaub Rhine gauge data")
    parser.add_argument("--bootstrap", action="store_true", help="fetch the full data.bs.ch history since 2020")
    parser.add_argument("--since", help="start date YYYY-MM-DD (default: last %d days)" % DAILY_LOOKBACK_DAYS)
    parser.add_argument("--no-s3", action="store_true", help="force local files even when S3 credentials exist")
    args = parser.parse_args()

    if args.no_s3:
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    today = date.today()
    if args.bootstrap:
        since = date.fromisoformat(BOOTSTRAP_START)
    elif args.since:
        since = date.fromisoformat(args.since)
    else:
        since = today - timedelta(days=DAILY_LOOKBACK_DAYS)
    until = today + timedelta(days=1)

    print(f"collecting {since}..{until} (bootstrap={args.bootstrap})")

    raw_frames = []
    for cfg in STATIONS:
        if cfg["bs_ch_dataset"]:
            raw_frames.append(fetch_bs_ch_history(cfg, since, until))

        # WSV only serves ~30 days of history: it is the only source for Kaub
        # and a cross-check for Basel.
        start_param = f"P{max((until - since).days, 1)}D"
        level = fetch_wsv_series(cfg["wsv_uuid"], "W", start=start_param)
        if not level.empty:
            level = level.rename(columns={"value": "level_cm"})
            level["station"] = cfg["station"]
            level["source"] = "wsv"
            raw_frames.append(level)
            print(f"  wsv {cfg['station']} W: {len(level)} rows")

        if cfg["wsv_discharge"]:
            discharge = fetch_wsv_series(cfg["wsv_uuid"], "Q", start=start_param)
            if not discharge.empty:
                discharge = discharge.rename(columns={"value": "discharge_m3s"})
                discharge["station"] = cfg["station"]
                discharge["source"] = "wsv"
                raw_frames.append(discharge)
                print(f"  wsv {cfg['station']} Q: {len(discharge)} rows")

    raw_frames = [frame for frame in raw_frames if not frame.empty]
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    daily = aggregate_daily(raw)
    print(f"daily aggregate: {len(daily)} rows")

    merged = merge_levels(load_csv(RHEIN_LEVELS_KEY, LOCAL_LEVELS_CSV), daily)
    save_csv(merged, RHEIN_LEVELS_KEY, LOCAL_LEVELS_CSV)

    snapshot = build_snapshot()
    save_json(snapshot, RHEIN_SNAPSHOT_KEY, LOCAL_SNAPSHOT_JSON)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

