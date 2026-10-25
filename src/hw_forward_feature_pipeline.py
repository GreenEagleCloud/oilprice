"""Forward-forecast feature pipeline for future target dates.

This pipeline creates a second Hopsworks feature group / feature view:

    heizoel24_forward_forecast

It is designed for predictions like:

    "Today is 2026-10-09. What is the expected oil price for the week after
     2026-11-11?"

The weekly feature view `heizoel24_weekly` only works when the requested date
already exists in the feature store, because its aggregate features are
computed from the price history up to that date.

The forward feature view instead stores rows keyed by:

    target_date + origin_date

where all model features are known at `origin_date`, and:

    forecast_horizon_days = target_date - origin_date

is a normal model feature.

For serving future dates, the inference app uses the latest available raw
price date as `origin_date` and looks up:

    target_date = requested future date
    origin_date = latest known origin date
"""

import sys

import numpy as np
import pandas as pd

from hsfs import feature_group

from hw_config import (
    DATA_FILE,
    DRIVER_CARRY_DAYS,
    FORWARD_EVENT_TIME,
    FORWARD_FEATURE_GROUP_NAME,
    FORWARD_FEATURE_GROUP_VERSION,
    FORWARD_FEATURES,
    FORWARD_FEATURE_VIEW_NAME,
    FORWARD_FEATURE_VIEW_VERSION,
    FORWARD_MARKET_FEATURES,
    FORWARD_ORIGIN_DATE_COLUMN,
    FORWARD_RHEIN_FEATURES,
    FORWARD_SERVING_HORIZONS,
    FORWARD_TRAINING_HORIZONS,
    FORWARD_PRIMARY_KEYS,
    FORWARD_TARGET,
    MARKET_FEATURES_CSV,
    MARKET_MODEL_FEATURES,
    RHEIN_LEVEL_FEATURES_CSV,
    RHEIN_MODEL_FEATURES,
    USE_MARKET_FEATURES,
    USE_RHEIN_FEATURES,
    forward_model_features,
    login,
)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Frühling"
    if month in (6, 7, 8):
        return "Sommer"
    return "Herbst"


def _load_raw_prices() -> pd.DataFrame:
    """Load raw prices from S3 when available, otherwise from the local CSV.

    In Kubernetes the fresh source of truth is the CSV written daily by
    `src/collect_daily_prices.py` to `s3://oilprice-features/raw/oil_data.csv`.
    The local `DATA_FILE` is only a fallback for local development or when S3
    credentials are unavailable.
    """
    try:
        from hw_feature_pipeline import load_prices

        df = load_prices()
    except Exception as e:
        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Raw price file not found: {DATA_FILE}. "
                "Run src/collect_daily_prices.py first."
            ) from e
        print(f"S3 raw price load failed ({e}); falling back to {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _seasonal_index(df: pd.DataFrame) -> pd.Series:
    """Average month-to-year price ratio, matching notebook 04."""
    ym = pd.DataFrame(
        {
            "year": df["date"].dt.year,
            "month": df["date"].dt.month,
            "price": df["price"].values,
        }
    )
    ym_mean = ym.groupby(["year", "month"])["price"].mean().reset_index()
    yearly_mean = ym.groupby("year")["price"].mean()
    ym_mean["ratio"] = ym_mean["price"] / ym_mean["year"].map(yearly_mean)
    return ym_mean.groupby("month")["ratio"].mean()


def _full_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Expand raw rows to a full daily calendar and forward-fill gaps.

    This mirrors `src/hw_feature_pipeline.py` so forward features use the same
    daily calendar as the weekly feature view.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").set_index("date")

    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    s = df.reindex(full)
    s.index.name = "date"
    s["price"] = s["price"].ffill()
    s["demand"] = s["demand"].ffill()
    return s.reset_index()


def _driver_frames() -> tuple[pd.DataFrame, list]:
    """Load the Rhine gauge / market driver features (written by their pipelines).

    Missing files are reported and filled with NULL: a forward row must still be
    created when a driver is unavailable, the model then sees a missing value.
    """
    frames, columns = [], []
    for label, path, cols, enabled in (
        ("Rhine gauge", RHEIN_LEVEL_FEATURES_CSV, RHEIN_MODEL_FEATURES, USE_RHEIN_FEATURES),
        ("oil market", MARKET_FEATURES_CSV, MARKET_MODEL_FEATURES, USE_MARKET_FEATURES),
    ):
        if not enabled:
            continue
        if path.exists():
            frame = pd.read_csv(path)
            frame["date"] = pd.to_datetime(frame["date"])
            missing = [col for col in cols if col not in frame.columns]
            if missing:
                print(f"WARNING: {label} driver file lacks {missing} - those columns will be NULL")
        else:
            print(f"WARNING: {label} driver file missing ({path}) - columns will be NULL")
            frame = pd.DataFrame({"date": pd.to_datetime([])})
        for col in cols:
            if col not in frame.columns:
                frame[col] = np.nan
        frames.append(frame[["date"] + list(cols)].drop_duplicates(subset="date", keep="last"))
        columns += list(cols)

    if not frames:
        return pd.DataFrame(), []
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    return merged, columns


def _attach_driver_columns(daily: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Merge the driver features onto the daily calendar and carry them forward.

    The market series only exist on trading days, so a weekend/Sunday price day
    must still carry Friday's market level. The carry is bounded to
    hw_config.DRIVER_CARRY_DAYS - the same rule the weekly feature pipeline uses -
    so a normal weekend gap is bridged but a data outage stays visible as a hole,
    and training and serving can never disagree about what a driver value means.
    """
    drivers, columns = _driver_frames()
    if not columns:
        return daily, []

    out = daily.merge(drivers, on="date", how="left")
    for col in columns:
        out[col] = out[col].ffill(limit=DRIVER_CARRY_DAYS)
    return out, columns


def _origin_features(df: pd.DataFrame, driver_cols: tuple = ()) -> pd.DataFrame:
    """Aggregate features known at each origin date."""
    out = df.copy()
    out = out.set_index("date")

    price = out["price"]
    demand = out["demand"]

    out["origin_price_mean_7d"] = price.rolling(7, min_periods=1).mean()
    out["origin_price_std_30d"] = price.rolling(30, min_periods=7).std()
    out["origin_demand_mean_7d"] = demand.rolling(7, min_periods=1).mean()
    out["origin_demand_mean_30d"] = demand.rolling(30, min_periods=7).mean()
    out["origin_price_mean_dow_4w"] = (
        price.shift(7) + price.shift(14) + price.shift(21) + price.shift(28)
    ) / 4
    out["origin_current_price"] = price
    out["origin_current_demand"] = demand

    # Driver features (Rhine gauge + oil market) are already point-in-time
    # values of the origin date, so they are only renamed with the origin_ prefix.
    for col in driver_cols:
        out[f"origin_{col}"] = out[col]

    return out.reset_index()


def _target_calendar_frame(target_dates: pd.Series, seasonal_map: dict) -> pd.DataFrame:
    month = target_dates.dt.month.astype(int)
    return pd.DataFrame(
        {
            "target_month": month,
            "target_season": month.map(_season),
            "target_month_sin": np.sin(2 * np.pi * month / 12).round(4),
            "target_month_cos": np.cos(2 * np.pi * month / 12).round(4),
            "target_seasonal_index": month.map(seasonal_map).fillna(1.0),
        }
    )


def build_forward_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build historical forward-forecast rows plus future serving rows.

    Historical rows have a non-null target label when the target week is
    already known. Future serving rows use the latest available origin date
    and have a null target label.
    """
    max_date = df["date"].max()
    daily = _full_daily_frame(df)
    daily, driver_cols = _attach_driver_columns(daily)
    origin = _origin_features(daily, driver_cols)
    seasonal_idx = _seasonal_index(daily)
    seasonal_map = {int(k): round(float(v), 4) for k, v in seasonal_idx.items()}

    origin_feature_cols = [
        "origin_price_mean_7d",
        "origin_price_std_30d",
        "origin_demand_mean_7d",
        "origin_demand_mean_30d",
        "origin_price_mean_dow_4w",
        "origin_current_price",
        "origin_current_demand",
    ]
    # Driver columns are copied too, but they are NOT part of the completeness
    # filter below: the Rhine series starts 2020 and the market series only on
    # trading days, so requiring them would delete most historical origins.
    driver_origin_cols = [f"origin_{col}" for col in driver_cols]
    copy_cols = origin_feature_cols + driver_origin_cols

    origin = origin.loc[origin[origin_feature_cols].notna().all(axis=1)].copy()
    if origin.empty:
        raise ValueError("No origin dates with complete forward features.")

    # Calendar-daily 7-day mean ending at date d equals mean(d-6..d).
    # For target_date t, the label is mean(t+1..t+7), i.e. ending at t+7.
    calendar_price = daily.set_index("date")["price"]
    calendar_mean_7d = calendar_price.rolling(7, min_periods=7).mean()

    frames = []

    # Historical training rows: origin_date is a real known date.
    for horizon in FORWARD_TRAINING_HORIZONS:
        target_dates = origin["date"] + pd.Timedelta(days=horizon)
        label_dates = target_dates + pd.Timedelta(days=7)
        labels = calendar_mean_7d.reindex(label_dates).to_numpy()
        valid = pd.notna(labels)

        if not valid.any():
            continue

        frame = pd.DataFrame(
            {
                "origin_date": origin.loc[valid, "date"],
                "target_date": target_dates.loc[valid],
                "forecast_horizon_days": int(horizon),
            }
        )
        for col in copy_cols:
            frame[col] = origin.loc[valid, col].to_numpy()

        calendar = _target_calendar_frame(target_dates.loc[valid], seasonal_map)
        for col in calendar.columns:
            frame[col] = calendar[col].to_numpy()

        frame[FORWARD_TARGET] = labels[valid]
        frames.append(frame)

    # Future serving rows: origin_date is the latest known raw-price date.
    latest_origin = origin.loc[origin["date"] == max_date]
    if not latest_origin.empty:
        target_dates = pd.Series(
            [max_date + pd.Timedelta(days=horizon) for horizon in FORWARD_SERVING_HORIZONS]
        )
        frame = pd.DataFrame(
            {
                "origin_date": max_date,
                "target_date": target_dates,
                "forecast_horizon_days": FORWARD_SERVING_HORIZONS,
            }
        )
        for col in copy_cols:
            frame[col] = latest_origin.iloc[0][col]

        calendar = _target_calendar_frame(target_dates, seasonal_map)
        for col in calendar.columns:
            frame[col] = calendar[col].to_numpy()

        frame[FORWARD_TARGET] = None
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise ValueError("No forward feature rows could be built.")

    out = out.sort_values(["target_date", "forecast_horizon_days"]).reset_index(drop=True)
    return out


def upsert_forward_feature_group(df: pd.DataFrame) -> feature_group.FeatureGroup:
    project = login()
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FORWARD_FEATURE_GROUP_NAME,
        version=FORWARD_FEATURE_GROUP_VERSION,
        description="Forward oil-price forecast features keyed by target_date + origin_date",
        primary_key=FORWARD_PRIMARY_KEYS,
        event_time=FORWARD_EVENT_TIME,
        online_enabled=True,
    )
    fg.insert(df, wait=True)
    print(
        f"Feature group '{FORWARD_FEATURE_GROUP_NAME}' v{FORWARD_FEATURE_GROUP_VERSION}: "
        f"{len(df)} rows upserted"
    )

    if not fg.online_enabled:
        try:
            fg.online_enabled = True
            print("Enabled online storage for the forward feature group.")
        except Exception as e:
            print(f"WARNING: could not enable online storage: {e}")

    return fg


def create_forward_feature_view(fg: feature_group.FeatureGroup):
    fs = fg.feature_store

    try:
        existing_fv = fs.get_feature_view(
            name=FORWARD_FEATURE_VIEW_NAME,
            version=FORWARD_FEATURE_VIEW_VERSION,
        )
        if existing_fv is not None:
            print(
                f"Feature view '{FORWARD_FEATURE_VIEW_NAME}' v{FORWARD_FEATURE_VIEW_VERSION} "
                "already exists."
            )
            return
    except Exception:
        pass

    # The view exposes only the level label. The stationary target the model
    # actually regresses (forward_target_week_change = FORWARD_TARGET -
    # origin_current_price) is derived in src/hw_forward_training_pipeline.py,
    # exactly like the weekly pipeline derives target_week_change: a stored delta
    # column would arrive in the feature view as a feature and could be handed to
    # the model as if it were an input.
    fv = fs.get_or_create_feature_view(
        name=FORWARD_FEATURE_VIEW_NAME,
        version=FORWARD_FEATURE_VIEW_VERSION,
        # `origin_date` is a primary key of the forward feature group and the
        # anchor the forward split is chronological over. Hopsworks leaves primary
        # keys out of a feature view unless they are selected, so it is selected
        # here; forward_model_features() still excludes it from the model matrix.
        # `target_date` stays unselected on purpose - origin_date +
        # forecast_horizon_days already determine it, so exposing it would only
        # add a leak surface.
        query=fg.select(
            [FORWARD_ORIGIN_DATE_COLUMN] + forward_model_features() + [FORWARD_TARGET]
        ),
        labels=[FORWARD_TARGET],
        description="Forward oil-price forecast: origin-date features (price/demand + Rhine/market drivers) + forecast_horizon_days -> target week after target_date",
    )

    print(
        f"Created feature view '{FORWARD_FEATURE_VIEW_NAME}' "
        f"v{fv.version} with primary keys {FORWARD_PRIMARY_KEYS}"
    )


def main() -> int:
    try:
        raw = _load_raw_prices()
        forward_df = build_forward_features(raw)

        fg = upsert_forward_feature_group(forward_df)
        create_forward_feature_view(fg)

        print("Forward feature pipeline finished.")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

