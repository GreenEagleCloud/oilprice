"""Hopsworks Feature Store pipeline for the heizoel24 oil-price data (TODO.md item 6).

Builds the same feature + label dataset as `notebooks/04_feature_engineering.ipynb`
(aggregate + real-time features + 4 target labels) from the raw price CSV on
S3 (written daily by src/collect_daily_prices.py; falls back to the local
repo copy when no S3 credentials are available), registers it as an
online-enabled feature group in Hopsworks, and creates a feature view that
selects only the 12 features + the training label.

Recent dates whose 7-day-ahead target is not known yet are stored with a NULL
label as well, so they can be served by /predict (the training pipeline
drops rows without a label).

Run:
    python src/hw_feature_pipeline.py
"""

import os

import numpy as np
import pandas as pd

from hw_config import (
    DATA_FILE,
    DATE_COLUMN,
    DRIVER_CARRY_DAYS,
    FEATURES,
    FEATURES_LOCAL_CSV,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    LABELS,
    MARKET_BASIS_FEATURES,
    MARKET_FEATURES_CSV,
    MARKET_MODEL_FEATURES,
    RHEIN_LEVEL_FEATURES_CSV,
    RHEIN_MODEL_FEATURES,
    TARGET,
    USE_MARKET_FEATURES,
    USE_RHEIN_FEATURES,
    login,
    weekly_model_features,
)


# Raw price data written daily by src/collect_daily_prices.py (source of truth)
RAW_S3_URI = "s3://oilprice-features/raw/oil_data.csv"


def load_prices() -> pd.DataFrame:
    """Read the raw price data (date, price, demand).

    Prefers the CSV written daily to S3 by src/collect_daily_prices.py (the
    live source of truth, kept fresh by the collect-daily-prices CronJob).
    Falls back to the repo copy baked into the image (DATA_FILE) when no S3
    credentials are set (e.g. local dev).
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        import boto3
        from io import BytesIO

        bucket, key = RAW_S3_URI[len("s3://"):].split("/", 1)
        s3 = boto3.client("s3")  # endpoint picked up from AWS_ENDPOINT_URL
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Loaded raw prices from {RAW_S3_URI}")
        return pd.read_csv(BytesIO(body))

    print(f"S3 credentials not available; using local copy {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature engineering of 04_feature_engineering.ipynb.

    Same steps as the notebook (without plots): extend to a full daily
    calendar + forward-fill, rolling aggregates, seasonal encoding,
    real-time features, target labels, drop NaN feature rows.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").set_index("date")

    # Auf vollen Tageskalender ausdehnen + fehlende Tage (WE/Feiertage) forward-fillen
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    s = df.reindex(full)
    s.index.name = "date"
    s["price"] = s["price"].ffill()
    s["demand"] = s["demand"].ffill()

    p = s["price"]

    # 1) 7-Tage-Mittelwert
    s["price_mean_7d"] = p.rolling(7, min_periods=1).mean()
    # 2) 30-Tage-Volatilität (Standardabweichung)
    s["price_std_30d"] = p.rolling(30, min_periods=7).std()
    # 3) Nachfrage (heizoel24) aggregiert
    s["demand_mean_7d"] = s["demand"].rolling(7, min_periods=1).mean()
    s["demand_mean_30d"] = s["demand"].rolling(30, min_periods=7).mean()

    # 4) Saisonaler Index + saisonale Kodierung
    month = s.index.month
    s["month"] = month
    s["season"] = month.map(
        {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Frühling", 4: "Frühling", 5: "Frühling",
            6: "Sommer", 7: "Sommer", 8: "Sommer",
            9: "Herbst", 10: "Herbst", 11: "Herbst",
        }
    )
    s["month_sin"] = np.sin(2 * np.pi * month / 12)
    s["month_cos"] = np.cos(2 * np.pi * month / 12)

    # trendrobuster saisonaler Index: (Ø Monat / Ø Jahr) pro (Jahr, Monat),
    # dann über alle Jahre gemittelt
    _ym = pd.DataFrame({"year": s.index.year, "month": s.index.month, "price": s["price"].values})
    _ym_mean = _ym.groupby(["year", "month"])["price"].mean().reset_index()
    _ym_mean["ratio"] = _ym_mean["price"] / _ym_mean["year"].map(_ym.groupby("year")["price"].mean())
    _seasonal_idx = _ym_mean.groupby("month")["ratio"].mean()
    s["seasonal_index"] = month.map(_seasonal_idx)

    # 5) Wochentags-Mittelwert: Ø desselben Wochentags, letzte 4 Wochen
    s["price_mean_dow_4w"] = (p.shift(7) + p.shift(14) + p.shift(21) + p.shift(28)) / 4

    # RT-Features: bekannt nur zur Inferenzzeit (Online Store); hier aus Historie rekonstruiert
    s["current_price"] = p            # Preis am Tag t
    s["current_demand"] = s["demand"]  # aktuelle heizoel24-Nachfrage am Tag t

    # Labels
    # Ø der nächsten 7 Tage: rolling(7) am Index t+7 = Ø(t+1..t+7); shift(-7) -> Ziel am Index t
    _rolling7 = p.rolling(7, min_periods=7).mean()
    s["target_next_week_price"] = _rolling7.shift(-7)
    s["target_week_change"] = s["target_next_week_price"] - s["current_price"]
    s["target_trend"] = (s["target_week_change"] > 0).astype(int)

    # konkreter Preis für einen spezifischen Monat (hier: nächster Kalendermonat)
    _month_avg = s["price"].groupby(s.index.to_period("M")).mean()
    s["target_next_month_price"] = (s.index.to_period("M") + 1).map(_month_avg)

    # Zeilen mit NaN in den FEATURES verwerfen (Start: 30d-Fenster + 4w-Wochentag).
    # Die letzten ~7 Tage haben (noch) kein 7d-Ziel-Label; die Zeilen werden trotzdem
    # gespeichert (NULL-Label), damit /predict fuer aktuelle Daten funktioniert.
    # Das Training filtert labellose Zeilen wieder heraus (hw_training_pipeline.py).
    dataset = s[FEATURES + LABELS].dropna(subset=FEATURES).reset_index()
    print(
        f"Feature+Label-Datensatz: {dataset.shape}, "
        f"Label bekannt fuer {int(dataset[TARGET].notna().sum())} Zeilen, "
        f"neuestes Datum: {dataset['date'].max().date()}"
    )
    return dataset


# ---------------------------------------------------------------------------
# External drivers (Rhine gauge + oil market)
# ---------------------------------------------------------------------------
def _read_driver_csv(path, columns: list, label: str) -> pd.DataFrame:
    """Read the driver feature CSV written by its own pipeline.

    Missing files or missing columns are reported and filled with NULL instead
    of failing the pipeline: the feature-group schema must stay identical from
    run to run, otherwise a later run could not upsert into the same group.
    """
    frame = pd.DataFrame({"date": pd.to_datetime([])})
    if not path.exists():
        print(f"WARNING: {label} driver file missing ({path}) - columns will be NULL")
    else:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            print(f"WARNING: {label} driver file has no 'date' column - columns will be NULL")
        else:
            df["date"] = pd.to_datetime(df["date"])
            missing = [col for col in columns if col not in df.columns]
            if missing:
                print(f"WARNING: {label} driver file lacks {missing} - those columns will be NULL")
            frame = df

    for col in columns:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[["date"] + list(columns)].drop_duplicates(subset="date", keep="last")


def load_driver_features() -> tuple[pd.DataFrame, list]:
    """Merge the Rhine gauge and oil-market driver features by date."""
    frames, columns = [], []
    for label, path, cols, enabled in (
        ("Rhine gauge", RHEIN_LEVEL_FEATURES_CSV, RHEIN_MODEL_FEATURES, USE_RHEIN_FEATURES),
        ("oil market", MARKET_FEATURES_CSV, MARKET_MODEL_FEATURES, USE_MARKET_FEATURES),
    ):
        if not enabled:
            print(f"{label} drivers disabled (OILPRICE_USE_*_FEATURES)")
            continue
        frames.append(_read_driver_csv(path, cols, label))
        columns += list(cols)

    if not frames:
        return pd.DataFrame(), []

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    return merged, columns


# Which market column each basis column is derived from. The names come from
# hw_config.MARKET_BASIS_FEATURES so the derived columns can never drift away
# from the list the feature view and the model contract select.
BASIS_SOURCES = {
    "market_basis_chf_100l": "market_price_chf_100l",
    "market_basis_chf_100l_mean_7d": "market_price_chf_100l_mean_7d",
}


def add_market_basis(frame: pd.DataFrame, price_col: str = "current_price") -> list:
    """Derive the stationary local-vs-market basis on the joined frame.

    `market_price_chf_100l` is a level that drifts across the training history;
    both models regress a CHANGE, and a RandomForest cannot extrapolate a level.
    The basis - what the Swiss shelf price costs above the landed market
    reference on the same day - is stationary around the local margin, so it is
    the form of the driver a tree can actually use.

    Point-in-time by construction: both operands belong to the row's own date.
    Returns the basis columns that were produced ([] when the market driver is
    disabled or absent).
    """
    if price_col not in frame.columns:
        return []
    produced = []
    for name in MARKET_BASIS_FEATURES:
        source = BASIS_SOURCES.get(name)
        if source and source in frame.columns:
            frame[name] = frame[source] - frame[price_col]
            produced.append(name)
    return produced


def attach_drivers(dataset: pd.DataFrame, drivers: pd.DataFrame, columns: list) -> pd.DataFrame:
    """LEFT-join the driver features onto the price rows and carry them forward.

    A driver that did not publish on a price day must neither delete the price
    row nor leave a hole the training pipeline drops. The market series exist on
    trading days only, so an exact-date join leaves ~107 NULL days a year, and
    requiring every driver non-null left the weekly model with 1 615 of 8 659
    labelled rows - the single largest reason it lost to persistence.

    The value of the most recent PAST publishing day is therefore carried for at
    most DRIVER_CARRY_DAYS days (a normal weekend/holiday gap) and stays NULL
    beyond that, so a real data outage stays visible as a hole instead of being
    papered over with a stale reading. Only past values are ever used, so the
    carry cannot leak the future into a training row.
    """
    if not columns:
        return dataset

    out = dataset.merge(drivers, on="date", how="left")
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan

    out = out.sort_values("date").reset_index(drop=True)
    for col in columns:
        out[col] = out[col].ffill(limit=DRIVER_CARRY_DAYS)

    basis = add_market_basis(out)
    # 'date' stays the first column: it is the primary key / event time of the
    # feature group and must survive the reordering.
    out = out[[
        "date", *FEATURES, *LABELS,
        *[c for c in columns if c in out.columns],
        *[c for c in basis if c not in columns],
    ]]

    coverage = ", ".join(f"{col}={int(out[col].notna().sum())}" for col in out.columns
                         if col in columns or col in basis)
    print(f"driver features joined: {len(columns)} columns + {len(basis)} derived "
          f"(carry <= {DRIVER_CARRY_DAYS}d) | coverage: {coverage}")
    return out


def main() -> int:
    dataset = build_dataset(load_prices())
    drivers, driver_columns = load_driver_features()
    dataset = attach_drivers(dataset, drivers, driver_columns)

    # Lokale Kopie (wie im Notebook: data/features/heizoel24_features.csv)
    FEATURES_LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(FEATURES_LOCAL_CSV, index=False)
    print(f"Gespeichert: {FEATURES_LOCAL_CSV}")

    project = login()
    fs = project.get_feature_store()

    # Feature Group: voller Datensatz (Features + alle 4 Labels), online-enabled
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="heizoel24 oil-price features + labels (notebook 04), online-enabled",
        online_enabled=True,
        primary_key=["date"],
        event_time="date",
    )
    # insert() is idempotent: it creates the FG metadata only if the FG does
    # not exist yet, and otherwise just validates the schema and upserts the
    # data (keyed by primary_key='date'). save() would re-POST the metadata and
    # fail with "already exists" on a re-run.
    fg.insert(dataset, wait=True)
    print(f"Feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}: {len(dataset)} rows upserted")

    # Feature View: model features + training label. The other labels from
    # Notebook 04 stay in the feature group so they can never leak into the
    # training features. The joined driver columns are part of the view.
    model_features = weekly_model_features()
    fv = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        # `date` is the primary key of the feature group and the anchor of both
        # the chronological split and the point-in-time seasonal index. Hopsworks
        # leaves primary keys out of a feature view unless they are selected, so
        # it is selected here; weekly_model_features() still excludes it from the
        # model matrix, and hw_app re-selects by the artifact's contract.
        query=fg.select([DATE_COLUMN] + model_features + [TARGET]),
        version=FEATURE_VIEW_VERSION,
        description=(
            "Weekly oil-price prediction: 12 price/demand features + "
            f"{len(model_features) - len(FEATURES)} Rhine/market driver features "
            "+ target_next_week_price"
        ),
        labels=[TARGET],
    )
    print(f"Feature view '{FEATURE_VIEW_NAME}' v{FEATURE_VIEW_VERSION} ready "
          f"({len(model_features)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
