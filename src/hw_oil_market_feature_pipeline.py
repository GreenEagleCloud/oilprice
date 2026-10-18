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
    LOCAL_MARKET_CSV,
    MARKET_FEATURES,
    MARKET_FEATURE_GROUP,
    MARKET_FEATURE_GROUP_VERSION,
    MARKET_FEATURE_VIEW,
    MARKET_FEATURE_VIEW_VERSION,
    MARKET_RAW_COLUMNS,
    MARKET_RAW_KEY,
    MARKET_SOURCE_COLUMNS,
    MAX_LAG_DAYS,
    MAX_WINDOW_SPAN_DAYS,
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

    The rolling windows are 7 *observations* (= one trading week), so they are
    only meaningful when those observations are actually one week apart. If the
    source has a hole - a discontinued series, a failed download - the 7th
    observation back can be months or years away, and the resulting "7-day
    change" is noise (a 16-year gap once produced +38.8 % here). Such rows get
    NULL instead of a bogus value, which the model then treats as missing.
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").set_index("date")

    price = df["market_price_chf_100l"].astype("float64")
    dates = df.index.to_series()

    # Calendar distance of the 7-observation window and of the 7-observation lag.
    window_span_days = (dates - dates.shift(6)).dt.days
    lag_span_days = (dates - dates.shift(7)).dt.days
    # NaN span only happens at the very start of the series, where the short
    # windows (min_periods) are still valid.
    window_ok = window_span_days.isna() | (window_span_days <= MAX_WINDOW_SPAN_DAYS)
    lag_ok = lag_span_days.isna() | (lag_span_days <= MAX_LAG_DAYS)

    mean_7d = price.rolling(7, min_periods=2).mean().where(window_ok)
    df["market_price_chf_100l_mean_7d"] = mean_7d.round(4)

    prev_price = price.shift(7).where(lag_ok).replace(0.0, np.nan)
    chg_7d = (price - prev_price) / prev_price * 100
    dropped = int((price.shift(7).notna() & ~lag_ok).sum())
    if dropped:
        print(
            f"  gap guard: dropped {dropped} 7-day change values whose 7-observation "
            f"lag exceeded {MAX_LAG_DAYS} calendar days (hole in the source series)"
        )
    df["market_price_chf_100l_chg_7d_pct"] = chg_7d.round(3)

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
