"""Hopsworks feature pipeline for the oil-market drivers (ICE Gasoil + USD/CHF).

Creates the feature group / feature view described in src/market_config.py:

    oil_market  primary key: date  event time: date  (online-enabled)
    oil_market_daily

Raw input is the CSV written by src/collect_oil_market_prices.py:
    s3://oilprice-features/raw/oil_market_prices.csv
    (fallback: data/oil/oil_market_prices.csv)

Point-in-time rules
-------------------
Every column of a row dated t is published on or before the close of t, so the
row may be used to predict the price of t and of any later date - never of an
earlier one. The 7-day smoothing windows only look backwards.

Run:
    python src/hw_oil_market_feature_pipeline.py                # build + upload
    python src/hw_oil_market_feature_pipeline.py --local-only   # CSV only
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_config import (  # noqa: E402
    FEATURES_LOCAL_CSV,
    MARKET_FEATURES,
    MARKET_FEATURE_GROUP,
    MARKET_FEATURE_GROUP_VERSION,
    MARKET_FEATURE_VIEW,
    MARKET_FEATURE_VIEW_VERSION,
    MARKET_RAW_COLUMNS,
    MARKET_SOURCE_COLUMNS,
    LOCAL_MARKET_CSV,
    MARKET_RAW_KEY,
)


def load_market_raw() -> pd.DataFrame:
    from collect_oil_market_prices import load_csv

    df = load_csv(MARKET_RAW_KEY, LOCAL_MARKET_CSV)
    if df is None or df.empty:
        raise FileNotFoundError(
            f"No market rows (s3 key {MARKET_RAW_KEY} / {LOCAL_MARKET_CSV}). "
            "Run src/collect_oil_market_prices.py first."
        )
    missing = [col for col in ("date", "gasoil_usd_tonne", "usd_chf") if col not in df.columns]
    if missing:
        raise ValueError(f"Market CSV is missing columns: {missing}")
    return df


def build_market_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Daily market features, one row per trading day.

    The raw collector already converts USD/t * USD/CHF into CHF/100l. On top of
    that the model gets a 7-day mean and the 7-day change in percent: the level
    carries the cost level, the change carries the momentum of the purchase
    price, which is what moves the Swiss retail price within a week.
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").set_index("date")

    # Trading days only: no weekend rows exist in the source series, so the
    # rolling windows are 7 *observations* (= one trading week).
    price = df["market_price_chf_100l"].astype("float64")
    df["market_price_chf_100l_mean_7d"] = price.rolling(7, min_periods=2).mean().round(4)
    df["market_price_chf_100l_chg_7d_pct"] = (
        (price - price.shift(7)) / price.shift(7).replace(0.0, np.nan) * 100
    ).round(3)

    out = df[MARKET_FEATURES + MARKET_SOURCE_COLUMNS].copy()
    out = out.dropna(subset=["gasoil_usd_tonne", "usd_chf", "market_price_chf_100l"])
    out = out.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    print(
        f"oil_market: {len(out)} rows {out['date'].min()} .. {out['date'].max()}, "
        f"sources gasoil={out['gasoil_source'].iloc[-1]} fx={out['fx_source'].iloc[-1]}"
    )
    return out


def upsert_feature_group(df: pd.DataFrame):
    from hw_config import login

    project = login()
    fs = project.get_feature_store()

    # Hopsworks requires the event time column to be a date/timestamp, but the
    # exported feature frame uses plain ISO date strings ("2026-10-18").
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fg = fs.get_or_create_feature_group(
        name=MARKET_FEATURE_GROUP,
        version=MARKET_FEATURE_GROUP_VERSION,
        description=(
            "Daily oil-market drivers: ICE Gasoil USD/t, USD/CHF and the derived "
            "CHF/100l purchase-price reference"
        ),
        primary_key=["date"],
        event_time="date",
        online_enabled=True,
    )
    # insert() upserts by primary key (see src/hw_feature_pipeline.py).
    fg.insert(df, wait=True)
    print(f"Feature group '{MARKET_FEATURE_GROUP}' v{MARKET_FEATURE_GROUP_VERSION}: {len(df)} rows upserted")

    if not fg.online_enabled:
        try:
            fg.online_enabled = True
            print(f"Enabled online storage for '{MARKET_FEATURE_GROUP}'.")
        except Exception as exc:
            print(f"WARNING: could not enable online storage: {exc}")
    return fg


def create_feature_view(fg, features):
    fs = fg.feature_store
    try:
        fs.get_feature_view(name=MARKET_FEATURE_VIEW, version=MARKET_FEATURE_VIEW_VERSION)
        print(f"Feature view '{MARKET_FEATURE_VIEW}' v{MARKET_FEATURE_VIEW_VERSION} already exists.")
        return
    except Exception:
        pass

    fs.get_or_create_feature_view(
        name=MARKET_FEATURE_VIEW,
        version=MARKET_FEATURE_VIEW_VERSION,
        query=fg.select(features),
        description="Daily ICE Gasoil / USD-CHF market features for the oil-price model",
    )
    print(f"Feature view '{MARKET_FEATURE_VIEW}' v{MARKET_FEATURE_VIEW_VERSION} ready")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the oil-market feature group")
    parser.add_argument("--local-only", action="store_true", help="write the CSV only, skip Hopsworks")
    args = parser.parse_args()

    features = build_market_features(load_market_raw())

    FEATURES_LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_LOCAL_CSV, index=False)
    print(f"saved {FEATURES_LOCAL_CSV} ({len(features)} rows)")

    if args.local_only:
        return 0

    fg = upsert_feature_group(features)
    create_feature_view(fg, MARKET_FEATURES)
    print("Market feature pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
