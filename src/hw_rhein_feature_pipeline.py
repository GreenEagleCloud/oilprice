"""Hopsworks feature pipeline for the Rhine gauge + Swiss Rhine-port cargo data.

Creates the two feature groups / feature views described in src/rhein_config.py:

    rhein_level  primary key: date    event time: date             (online-enabled)
    rhein_cargo  primary key: month   event time: published_date

Raw inputs are the CSVs written by the collectors:
    src/collect_rhein_levels.py -> s3://oilprice-features/raw/rhein_levels.csv
    src/collect_rhein_cargo.py  -> s3://oilprice-features/raw/rhein_cargo.csv
    (fallback: data/rhein/rhein_levels.csv / data/rhein/rhein_cargo.csv)

Point-in-time rules
-------------------
* Gauge features only use readings up to and including their own date.
* Cargo rows carry the publication date of the bulletin as event time. The
  numbers describe the reporting month, which is published 4-7 weeks later, so
  they are LAGGED features: a training query must join them onto the price
  table with a lag (>= 49 days), never as same-day information.

Run:
    python src/hw_rhein_feature_pipeline.py                # build + upload
    python src/hw_rhein_feature_pipeline.py --local-only   # CSVs only
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rhein_config import (  # noqa: E402
    BASEL,
    CARGO_MAIN_GROUPS,
    CARGO_TOTAL_GROUP,
    KAUB,
    LOCAL_CARGO_CSV,
    LOCAL_LEVELS_CSV,
    RHEIN_CARGO_FEATURES,
    RHEIN_CARGO_FEATURE_GROUP,
    RHEIN_CARGO_FEATURE_GROUP_VERSION,
    RHEIN_CARGO_FEATURE_VIEW,
    RHEIN_CARGO_FEATURE_VIEW_VERSION,
    RHEIN_CARGO_FEATURES_CSV,
    RHEIN_CARGO_KEY,
    RHEIN_LEVEL_FEATURES,
    RHEIN_LEVEL_FEATURE_GROUP,
    RHEIN_LEVEL_FEATURE_GROUP_VERSION,
    RHEIN_LEVEL_FEATURE_VIEW,
    RHEIN_LEVEL_FEATURE_VIEW_VERSION,
    RHEIN_LEVEL_FEATURES_CSV,
    RHEIN_LEVELS_KEY,
)


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
def load_levels() -> pd.DataFrame:
    from collect_rhein_levels import load_csv

    df = load_csv(RHEIN_LEVELS_KEY, LOCAL_LEVELS_CSV)
    if df is None or df.empty:
        raise FileNotFoundError(
            f"No Rhine gauge rows (s3 key {RHEIN_LEVELS_KEY} / {LOCAL_LEVELS_CSV}). "
            "Run src/collect_rhein_levels.py first."
        )
    return df


def load_cargo() -> pd.DataFrame:
    from collect_rhein_cargo import load_cargo as load_raw

    df = load_raw()
    if df is None or df.empty:
        raise FileNotFoundError(
            f"No cargo rows (s3 key {RHEIN_CARGO_KEY} / {LOCAL_CARGO_CSV}). "
            "Run src/collect_rhein_cargo.py first."
        )
    return df


# ---------------------------------------------------------------------------
# Daily gauge features
# ---------------------------------------------------------------------------
def _station_frame(levels: pd.DataFrame, station: str) -> pd.DataFrame:
    frame = levels[levels["station"] == station].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates(subset="date", keep="last")
    return frame.set_index("date")


def _attach(df: pd.DataFrame, frame: pd.DataFrame, prefix: str, low_cm: float) -> None:
    """Add <prefix>_level_cm plus the rolling features of one station."""
    if frame.empty:
        for name in ("level_cm", "level_min_7d", "level_mean_7d", "level_zscore_30d",
                     "low_days_30d", "discharge_m3s"):
            df[f"{prefix}_{name}"] = np.nan
        return

    # The daily MINIMUM is the shipping-relevant value: a tanker on its way to
    # Basel is limited by the lowest reading of the day at the bottleneck.
    level = frame["level_cm_min"].astype("float64").reindex(df.index)
    level = level.ffill(limit=2)  # a single missing gauge day stays usable
    df[f"{prefix}_level_cm"] = level
    df[f"{prefix}_level_min_7d"] = level.rolling(7, min_periods=4).min()
    df[f"{prefix}_level_mean_7d"] = level.rolling(7, min_periods=4).mean()

    mean30 = level.rolling(30, min_periods=15).mean()
    std30 = level.rolling(30, min_periods=15).std()
    df[f"{prefix}_level_zscore_30d"] = ((level - mean30) / std30.replace(0.0, np.nan)).round(3)

    low = (level < low_cm).astype("float64").where(level.notna())
    df[f"{prefix}_low_days_30d"] = low.rolling(30, min_periods=15).sum()

    if "discharge_m3s_mean" in frame.columns:
        df[f"{prefix}_discharge_m3s"] = (
            frame["discharge_m3s_mean"].astype("float64").reindex(df.index).ffill(limit=2)
        )
    else:
        df[f"{prefix}_discharge_m3s"] = np.nan

def build_level_features(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily Rhine gauge features, one row per UTC calendar day."""
    basel = _station_frame(levels, BASEL["station"])
    kaub = _station_frame(levels, KAUB["station"])
    if basel.empty:
        raise ValueError("No Basel gauge rows; run src/collect_rhein_levels.py first.")

    last = max(basel.index.max(), kaub.index.max() if not kaub.empty else basel.index.max())
    df = pd.DataFrame(index=pd.date_range(basel.index.min(), last, freq="D"))
    df.index.name = "date"

    _attach(df, basel, "basel", BASEL["low_water_cm"])
    _attach(df, kaub, "kaub", KAUB["low_water_cm"])

    # min(Basel, Kaub): a Basel-bound tanker has to pass both gauges, so the
    # lower one decides how much it can load. Missing Kaub readings (WSV only
    # serves the last ~30 days) fall back to Basel, i.e. an upper bound.
    df["bottleneck_level_cm"] = df[["basel_level_cm", "kaub_level_cm"]].min(axis=1, skipna=True)
    df["bottleneck_level_min_7d"] = df["bottleneck_level_cm"].rolling(7, min_periods=4).min()

    # 7-day trend: 7-day mean against the 7-day mean one week earlier
    df["basel_level_trend_7d"] = (df["basel_level_mean_7d"] - df["basel_level_mean_7d"].shift(7)).round(2)

    low_basel = (df["basel_level_cm"] < BASEL["low_water_cm"]).fillna(False)
    low_kaub = (df["kaub_level_cm"] < KAUB["low_water_cm"]).fillna(False)
    df["rhein_low_water_flag"] = (low_basel | low_kaub).astype(int)

    df = df.dropna(subset=["basel_level_cm"])
    out = df[RHEIN_LEVEL_FEATURES].round(2).reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    print(
        f"rhein_level: {len(out)} rows {out['date'].min()} .. {out['date'].max()}, "
        f"Kaub covered for {int(out['kaub_level_cm'].notna().sum())} days"
    )
    return out


# ---------------------------------------------------------------------------
# Monthly cargo features
# ---------------------------------------------------------------------------
# feature column <- (goods group, direction)
CARGO_SOURCE_COLUMNS = {
    "oil_zufuhr_t": (CARGO_MAIN_GROUPS, "zufuhr"),
    "oil_abfuhr_t": (CARGO_MAIN_GROUPS, "abfuhr"),
    "gas_diesel_heizoel_zufuhr_t": (("07.23",), "zufuhr"),
    "gas_diesel_heizoel_abfuhr_t": (("07.23",), "abfuhr"),
    "motorbenzin_zufuhr_t": (("07.21",), "zufuhr"),
    "jet_petroleum_zufuhr_t": (("07.22",), "zufuhr"),
    "lubricants_abfuhr_t": (("07.25",), "abfuhr"),
    "cargo_total_zufuhr_t": ((CARGO_TOTAL_GROUP,), "zufuhr"),
    "cargo_total_abfuhr_t": ((CARGO_TOTAL_GROUP,), "abfuhr"),
}


def build_cargo_features(cargo: pd.DataFrame) -> pd.DataFrame:
    """Monthly cargo features, one row per bulletin month.

    The month is expanded to a full calendar so MoM/YoY changes compare the
    right months even when a bulletin is missing from the archive.
    """
    data = cargo.copy()
    data["month"] = pd.PeriodIndex(pd.to_datetime(data["month"] + "-01"), freq="M")

    wide = pd.DataFrame(index=pd.period_range(data["month"].min(), data["month"].max(), freq="M"))
    for column, (groups, direction) in CARGO_SOURCE_COLUMNS.items():
        selected = data[data["group"].isin(groups) & (data["direction"] == direction)]
        if selected.empty:
            wide[column] = np.nan
            continue
        values = selected.groupby("month")["tonnes"].min()
        wide[column] = values.reindex(wide.index).astype("float64")

    meta = data.sort_values("published_date").groupby("month").last()
    wide["published_date"] = meta["published_date"].reindex(wide.index)
    wide["oil_zufuhr_ytd_t"] = (
        data[data["group"].isin(CARGO_MAIN_GROUPS) & (data["direction"] == "zufuhr")]
        .groupby("month")["current_ytd_tonnes"].max()
        .reindex(wide.index)
    )

    wide["oil_zufuhr_prev_month_t"] = wide["oil_zufuhr_t"].shift(1)
    wide["oil_zufuhr_mom_pct"] = (
        (wide["oil_zufuhr_t"] - wide["oil_zufuhr_prev_month_t"])
        / wide["oil_zufuhr_prev_month_t"].replace(0.0, np.nan)
        * 100
    ).round(2)
    prev_year = wide["oil_zufuhr_t"].shift(12)
    wide["oil_zufuhr_yoy_pct"] = (
        (wide["oil_zufuhr_t"] - prev_year) / prev_year.replace(0.0, np.nan) * 100
    ).round(2)

    out = wide[RHEIN_CARGO_FEATURES + ["published_date"]].copy()
    # Hopsworks requires a timestamp column for the event time
    out["published_date"] = pd.to_datetime(out["published_date"])
    out["month"] = [period.strftime("%Y-%m") for period in wide.index]
    out = out.reset_index(drop=True).sort_values("month").reset_index(drop=True)
    known = out["oil_zufuhr_t"].notna().sum()
    print(
        f"rhein_cargo: {len(out)} months {out['month'].min()} .. {out['month'].max()}, "
        f"oil turnover known for {known} months"
    )
    return out



# ---------------------------------------------------------------------------
# Hopsworks
# ---------------------------------------------------------------------------
def upsert_feature_group(df: pd.DataFrame, name: str, version: int, primary_key: list,
                         event_time: str, description: str):
    from hw_config import login

    project = login()
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time=event_time,
        online_enabled=True,
    )
    # insert() upserts by primary key; save() would re-POST the metadata and
    # fail with "already exists" on a re-run (same pattern as hw_feature_pipeline).
    fg.insert(df, wait=True)
    print(f"Feature group '{name}' v{version}: {len(df)} rows upserted")

    if not fg.online_enabled:
        try:
            fg.online_enabled = True
            print(f"Enabled online storage for '{name}'.")
        except Exception as exc:
            print(f"WARNING: could not enable online storage: {exc}")
    return fg


def create_feature_view(fg, name: str, version: int, features: list, description: str):
    fs = fg.feature_store
    try:
        fs.get_feature_view(name=name, version=version)
        print(f"Feature view '{name}' v{version} already exists.")
        return
    except Exception:
        pass

    fs.get_or_create_feature_view(
        name=name,
        version=version,
        query=fg.select(features),
        description=description,
    )
    print(f"Feature view '{name}' v{version} ready")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Rhine feature groups")
    parser.add_argument("--local-only", action="store_true",
                        help="write the feature CSVs only, skip Hopsworks")
    args = parser.parse_args()

    levels = build_level_features(load_levels())
    cargo = build_cargo_features(load_cargo())

    RHEIN_LEVEL_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    levels.to_csv(RHEIN_LEVEL_FEATURES_CSV, index=False)
    cargo.to_csv(RHEIN_CARGO_FEATURES_CSV, index=False)
    print(f"saved {RHEIN_LEVEL_FEATURES_CSV} ({len(levels)} rows)")
    print(f"saved {RHEIN_CARGO_FEATURES_CSV} ({len(cargo)} rows)")

    if args.local_only:
        return 0

    level_fg = upsert_feature_group(
        levels,
        name=RHEIN_LEVEL_FEATURE_GROUP,
        version=RHEIN_LEVEL_FEATURE_GROUP_VERSION,
        primary_key=["date"],
        event_time="date",
        description="Daily Rhine gauge features (Basel-Rheinhalle + Kaub bottleneck)",
    )
    create_feature_view(
        level_fg,
        RHEIN_LEVEL_FEATURE_VIEW,
        RHEIN_LEVEL_FEATURE_VIEW_VERSION,
        RHEIN_LEVEL_FEATURES,
        "Daily Rhine low-water features for the heating-oil price model",
    )

    cargo_fg = upsert_feature_group(
        cargo,
        name=RHEIN_CARGO_FEATURE_GROUP,
        version=RHEIN_CARGO_FEATURE_GROUP_VERSION,
        primary_key=["month"],
        event_time="published_date",
        description="Monthly Swiss Rhine port cargo (mineral oil) features, published with a 4-7 week lag",
    )
    create_feature_view(
        cargo_fg,
        RHEIN_CARGO_FEATURE_VIEW,
        RHEIN_CARGO_FEATURE_VIEW_VERSION,
        RHEIN_CARGO_FEATURES,
        "Lagged monthly oil turnover of the Port of Switzerland (Basel)",
    )
    print("Rhine feature pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

