"""Build and maintain a Kaub (Middle-Rhine bottleneck) gauge history.

Why a separate archive is needed
--------------------------------
Kaub is the gauge that limits how deep a Basel-bound tanker can load, and WSV
Pegelonline is the only free source for it - but the WSV REST API serves just
the last ~30 days. Verified live 2026-10-18:

    GET /stations/<kaub>/W/measurements.json?start=2020-01-01
        -> first row 2026-09-17 (the request is clamped to the last 30 days)
    GET /stations/<kaub>/W/measurements.json?start=2020-01-01&end=2021-01-01
        -> []            (older windows are simply empty)
    GET /stations/<kaub>/W/measurements.json?start=2025-01-01&end=2025-02-01
        -> []

There is no public multi-year Kaub API, so the history has to be accumulated
by us. Every run appends the currently available window to an append-only
archive; after a few months the archive is longer than the WSV window and the
daily Kaub features no longer depend on a 30-day horizon.

Outputs
-------
    s3://oilprice-features/raw/rhein_kaub_raw.csv   append-only raw archive
        (fallback: data/rhein/rhein_kaub_raw.csv)
    s3://oilprice-features/raw/rhein_levels.csv     daily Kaub rows (merged)

Run:
    python src/collect_rhein_kaub_history.py              # daily update
    python src/collect_rhein_kaub_history.py --rebuild    # re-aggregate archive
    python src/collect_rhein_kaub_history.py --no-s3
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rhein_config import (  # noqa: E402
    KAUB,
    KAUB_ARCHIVE_WINDOW_DAYS,
    LOCAL_KAUB_RAW_CSV,
    LOCAL_LEVELS_CSV,
    RHEIN_KAUB_RAW_KEY,
    RHEIN_LEVELS_KEY,
)

RAW_COLUMNS = ["timestamp_utc", "station", "source", "level_cm", "discharge_m3s"]


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
def load_archive():
    from collect_rhein_levels import load_csv

    return load_csv(RHEIN_KAUB_RAW_KEY, LOCAL_KAUB_RAW_CSV)


def append_archive(existing, new) -> pd.DataFrame:
    """Append-only merge keyed by timestamp (WSV corrects readings late)."""
    frames = [frame for frame in (existing, new) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], utc=True, format="ISO8601")
    merged = merged.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    merged["timestamp_utc"] = merged["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return merged[RAW_COLUMNS].reset_index(drop=True)


def fetch_window(days):
    """Raw Kaub level + discharge readings of the currently available window."""
    from collect_rhein_levels import fetch_wsv_series

    level = fetch_wsv_series(KAUB["wsv_uuid"], "W", start=f"P{max(int(days), 1)}D")
    if level.empty:
        print("WARNING: WSV returned no Kaub level readings")
        return pd.DataFrame(columns=RAW_COLUMNS)
    level = level.rename(columns={"value": "level_cm"})

    discharge = fetch_wsv_series(KAUB["wsv_uuid"], "Q", start=f"P{max(int(days), 1)}D")
    if not discharge.empty:
        discharge = discharge.rename(columns={"value": "discharge_m3s"})
        level = level.merge(discharge, on="timestamp", how="left")
    else:
        level["discharge_m3s"] = None

    level = level.rename(columns={"timestamp": "timestamp_utc"})
    level["station"] = KAUB["station"]
    level["source"] = "wsv"
    print(f"  wsv {KAUB['station']} window P{int(days)}D: {len(level)} raw readings")
    return level[RAW_COLUMNS]


def refresh_daily_levels(archive):
    """Aggregate the whole archive into the Kaub rows of raw/rhein_levels.csv."""
    from collect_rhein_levels import aggregate_daily, load_csv, merge_levels, save_csv

    if archive is None or archive.empty:
        return pd.DataFrame()

    raw = archive.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp_utc"], utc=True, format="ISO8601")
    raw["level_cm"] = pd.to_numeric(raw["level_cm"], errors="coerce")
    raw["discharge_m3s"] = pd.to_numeric(raw["discharge_m3s"], errors="coerce")

    daily = aggregate_daily(raw)
    merged = merge_levels(load_csv(RHEIN_LEVELS_KEY, LOCAL_LEVELS_CSV), daily)
    save_csv(merged, RHEIN_LEVELS_KEY, LOCAL_LEVELS_CSV)

    kaub = merged[merged["station"] == KAUB["station"]]
    if len(kaub):
        print(
            f"kaub daily rows: {len(kaub)} {kaub['date'].min()} .. {kaub['date'].max()}"
        )
    return daily


def main() -> int:
    parser = argparse.ArgumentParser(description="Maintain the Kaub gauge history archive")
    parser.add_argument("--window-days", type=int, default=KAUB_ARCHIVE_WINDOW_DAYS,
                        help="WSV window to fetch (WSV caps history at ~30 days)")
    parser.add_argument("--rebuild", action="store_true",
                        help="skip the fetch and re-aggregate the stored archive")
    parser.add_argument("--no-s3", action="store_true", help="force local files")
    args = parser.parse_args()

    if args.no_s3:
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    if args.rebuild:
        archive = load_archive()
        if archive is None or archive.empty:
            print("ERROR: no Kaub archive to rebuild", file=sys.stderr)
            return 1
    else:
        archive = append_archive(load_archive(), fetch_window(args.window_days))
        if archive.empty:
            print("ERROR: no Kaub readings collected", file=sys.stderr)
            return 1

    from collect_rhein_levels import save_csv

    save_csv(archive, RHEIN_KAUB_RAW_KEY, LOCAL_KAUB_RAW_CSV)
    print(
        f"kaub archive: {len(archive)} raw readings "
        f"{archive['timestamp_utc'].min()} .. {archive['timestamp_utc'].max()}"
    )

    refresh_daily_levels(archive)
    print(f"collected_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

