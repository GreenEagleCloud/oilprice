import json
import math
import os
import time
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend service URL (overridable via the BACKEND_URL env var, set in the
# k8s frontend deployment). Defaults to the in-cluster inference service.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://inference-service:5000/predict")

# Raw price data written daily by src/collect_daily_prices.py (source of truth).
PRICE_S3_URI = os.environ.get("PRICE_S3_URI", "s3://oilprice-features/raw/oil_data.csv")
PRICE_LOCAL_FILE = os.environ.get("PRICE_LOCAL_FILE", "")  # empty = auto-detect
_PRICE_FILENAME = "heizoel24_heizoel_daily_ch_full.csv"


def _local_price_candidates() -> list:
    base = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(base, "data", _PRICE_FILENAME),  # docker image: /app/data/...
        os.path.abspath(os.path.join(base, "..", "..", "data", _PRICE_FILENAME)),  # local dev: repo root
    ]
# How long to cache the price history between S3/local reads (seconds).
PRICE_CACHE_TTL = int(os.environ.get("PRICE_CACHE_TTL", 15 * 60))

# ---------------------------------------------------------------------------
# Readable labels for the "Vom Modell verwendete Merkmale" table.
#
# The names mirror the model feature lists in src/hw_config.py (FEATURES +
# RHEIN_MODEL_FEATURES + MARKET_MODEL_FEATURES, and the origin_/target_
# variants of FORWARD_FEATURES). They are repeated here because the frontend
# image contains only src/frontend and cannot import src/hw_config.py - same
# reason the low-water thresholds above are duplicated.
# tests/test_frontend_ux.py checks that every model feature has a label.
# ---------------------------------------------------------------------------
_CORE_FEATURE_LABELS = {
    # price & demand of heizoel24.ch
    "price_mean_7d": ("Preis-Durchschnitt (7 Tage)", "CHF/100l", "Preis & Nachfrage"),
    "price_std_30d": ("Preisschwankung (30 Tage)", "CHF", "Preis & Nachfrage"),
    "demand_mean_7d": ("Nachfrage-Durchschnitt (7 Tage)", "Index", "Preis & Nachfrage"),
    "demand_mean_30d": ("Nachfrage-Durchschnitt (30 Tage)", "Index", "Preis & Nachfrage"),
    "current_price": ("Aktueller Preis", "CHF/100l", "Preis & Nachfrage"),
    "current_demand": ("Nachfrageindex aktuell", "Index", "Preis & Nachfrage"),
    "price_mean_dow_4w": ("Durchschnittspreis gleicher Wochentag (4 Wochen)", "CHF/100l", "Preis & Nachfrage"),
    # calendar / seasonality
    "month": ("Monat", "", "Zeit & Saison"),
    "season": ("Jahreszeit", "", "Zeit & Saison"),
    "month_sin": ("Monat zyklisch (Sinus)", "", "Zeit & Saison"),
    "month_cos": ("Monat zyklisch (Kosinus)", "", "Zeit & Saison"),
    "seasonal_index": ("Saisonaler Index", "", "Zeit & Saison"),
    "target_month": ("Monat des Zieldatums", "", "Zeit & Saison"),
    "target_season": ("Jahreszeit des Zieldatums", "", "Zeit & Saison"),
    "target_month_sin": ("Zieldatum zyklisch (Sinus)", "", "Zeit & Saison"),
    "target_month_cos": ("Zieldatum zyklisch (Kosinus)", "", "Zeit & Saison"),
    "target_seasonal_index": ("Saisonaler Index des Zieldatums", "", "Zeit & Saison"),
    "forecast_horizon_days": ("Prognosehorizont ab Ausgangstag", "Tage", "Zeit & Saison"),
    # Rhine gauge drivers
    "basel_level_cm": ("Rheinpegel Basel-Rheinhalle (Tagesminimum)", "cm", "Rhein-Pegel"),
    "basel_level_min_7d": ("Rheinpegel Basel, tiefster Wert (7 Tage)", "cm", "Rhein-Pegel"),
    "basel_level_zscore_30d": ("Rheinpegel Basel, Abweichung vom 30-Tage-Mittel", "Standardabweichungen", "Rhein-Pegel"),
    "basel_low_days_30d": ("Niedrigwassertage Basel (30 Tage)", "Tage", "Rhein-Pegel"),
    "basel_level_trend_7d": ("Pegeltrend Basel (7 Tage)", "cm", "Rhein-Pegel"),
    "kaub_level_cm": ("Rheinpegel Kaub (Tagesminimum)", "cm", "Rhein-Pegel"),
    "rhein_low_water_flag": ("Niedrigwasser-Alarm (Rhein)", "0 / 1", "Rhein-Pegel"),
    # oil market drivers
    "market_price_chf_100l": ("Ölmarkt-Referenzpreis (Gasoil zu CHF umgerechnet)", "CHF/100l", "Ölmarkt"),
    "market_price_chf_100l_mean_7d": ("Ölmarkt-Referenzpreis-Durchschnitt (7 Tage)", "CHF/100l", "Ölmarkt"),
    "market_price_chf_100l_chg_7d_pct": ("Ölmarkt-Referenzpreis-Änderung (7 Tage)", "%", "Ölmarkt"),
    "gasoil_usd_tonne": ("ICE-Gasoil-Futures", "USD/Tonne", "Ölmarkt"),
    "usd_chf": ("USD/CHF-Wechselkurs", "CHF/USD", "Ölmarkt"),
}

# The forward model keys every driver by origin_date, so those columns carry an
# "origin_" prefix; the label says which date the value belongs to.
_ORIGIN_LABELS = {
    f"origin_{name}": (f"Ausgangstag: {label}", unit, group)
    for name, (label, unit, group) in _CORE_FEATURE_LABELS.items()
    if name not in ("month", "season", "month_sin", "month_cos", "seasonal_index")
}

FEATURE_LABELS = {
    name: {"label": label, "unit": unit, "group": group}
    for name, (label, unit, group) in {**_CORE_FEATURE_LABELS, **_ORIGIN_LABELS}.items()
}

_price_cache = {"ts": 0.0, "data": None}


def _read_price_history() -> tuple:
    """Load the heizoel24 daily price/demand series.

    Prefers the CSV written daily to S3 by the collect CronJob (fresh
    source of truth); falls back to the CSV bundled into the image
    (see Dockerfile.frontend) when S3 is not configured or reachable.

    Returns (series_dict, source) where source is 's3' or 'local'.
    """
    bucket, key = PRICE_S3_URI[len("s3://"):].split("/", 1)
    try:
        s3 = boto3.client("s3")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_csv(BytesIO(body))
        source = "s3"
    except Exception as e:
        print(f"warning: S3 read failed ({e}); falling back to bundled CSV")
        candidates = [PRICE_LOCAL_FILE] if PRICE_LOCAL_FILE else _local_price_candidates()
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            raise RuntimeError(f"no local price CSV found (tried {candidates})")
        df = pd.read_csv(path)
        source = "local"

    # normalize column names (collect writes date/price/demand; the 5y file
    # uses price_chf_per_100l)
    rename = {}
    if "price" not in df.columns:
        for col in df.columns:
            if col.lower() in ("price", "price_chf_per_100l", "preis"):
                rename[col] = "price"
                break
    if "demand" not in df.columns:
        for col in df.columns:
            if col.lower().startswith("demand"):
                rename[col] = "demand"
                break
    df = df.rename(columns=rename)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    prices = df["price"].astype(float).where(df["price"].notna(), None)
    demands = (
        df["demand"].astype(float).where(df["demand"].notna(), None)
        if "demand" in df.columns
        else [None] * len(df)
    )
    return (
        {
            "dates": [str(d) for d in df["date"]],
            "price": [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in prices],
            "demand": [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in demands],
        },
        source,
    )


# ---------------------------------------------------------------------------
# Forecast curve drawn on the price chart
#
# The price chart shows the measured history and, from the last known price
# date on, a dotted forecast built from BOTH models:
#   * weekly model  -> the first week after the last known price date
#   * forward model -> the following weeks out to FORECAST_MAX_HORIZON days
#
# Both models answer "what is the average price of the week AFTER the date you
# ask about" (weekly: target_next_week_price, forward:
# target_price_week_after_target_date). A point plotted at the requested date
# would therefore sit a week too early, so every forecast point is placed at
# the middle of the week it describes - see _forecast_week_midpoint().
#
# The curve is assembled here rather than in the browser because it needs one
# /predict request per horizon step; doing that client-side would fire ~10
# requests and repeat all of them on every price/demand edit.
#
# The response also carries the measured daily prices of the forecast window
# ("measured", one entry per calendar day). The history line must not stop at
# the origin date: as soon as a forecasted day is measured, the real price is
# drawn through the forecast region so prediction and reality are comparable.
# ---------------------------------------------------------------------------
# The chart cone is drawn out to three months: the forward model serves up to
# 90 days, and the horizon steps are weekly, so the last point lands at 84 days.
FORECAST_MAX_HORIZON = int(os.environ.get("FORECAST_MAX_HORIZON", 90))
FORECAST_CACHE_TTL = int(os.environ.get("FORECAST_CACHE_TTL", 10 * 60))
WEEKLY_HORIZON_DAYS = 7  # the weekly model always answers for the next 7 days
# Horizon steps asked of the forward model (its feature view serves 1..90 days).
FORWARD_HORIZON_STEPS = (7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84)
FORWARD_MAX_SERVING_HORIZON = 90
# The models return a point prediction only - no uncertainty. The band is
# therefore an explicitly labelled volatility estimate, not a model confidence
# interval: the standard deviation of the 7-day price change over the last
# VOLATILITY_DAYS days, growing with sqrt(horizon) the way an unguarded random
# walk does (measured on the real series, the empirical horizon volatility
# follows that sqrt curve closely).
VOLATILITY_DAYS = int(os.environ.get("FORECAST_VOLATILITY_DAYS", 3 * 365))
BAND_SIGMA = float(os.environ.get("FORECAST_BAND_SIGMA", 1.0))

_forecast_cache = {"ts": 0.0, "key": None, "data": None}


def _post_predict(payload: dict) -> tuple:
    """One prediction request to the inference service. Returns (payload, status)."""
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=30)
    except requests.RequestException as e:
        return {"error": f"prediction service is unreachable: {e}"}, 502
    try:
        body = response.json()
    except ValueError:
        return {
            "error": f"prediction service returned a non-JSON response (HTTP {response.status_code})",
        }, 502
    return body, response.status_code


def _forecast_week_midpoint(date_str: str) -> str:
    """Middle of the week a prediction describes (the week after date_str)."""
    return (pd.Timestamp(date_str) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")


def _last_price_point(series: dict):
    """(date, price) of the newest day that really has a price - the forecast origin."""
    dates = series.get("dates") or []
    prices = series.get("price") or []
    for i in range(len(dates) - 1, -1, -1):
        value = prices[i] if i < len(prices) else None
        if value is not None:
            return str(dates[i]), float(value)
    return None


def _measured_prices(series: dict, start_date: str, end_date: str) -> list:
    """Real daily prices between two dates - one entry per calendar day.

    A forecast point describes a whole week that lies ahead, but the days of
    that week that are already measured must appear in the chart too: without
    them the history line stops exactly where the prediction begins and the two
    can never be compared in one view. Days the source has no price for are
    returned with value None, so the chart keeps one label per day and the
    measured line simply has a hole where nothing was published.
    """
    dates = series.get("dates") or []
    prices = series.get("price") or []
    known = {str(d): v for d, v in zip(dates, prices) if v is not None}
    out = []
    day, last = pd.Timestamp(start_date), pd.Timestamp(end_date)
    while day <= last:
        key = day.strftime("%Y-%m-%d")
        value = known.get(key)
        out.append({"date": key, "value": round(float(value), 2) if value is not None else None})
        day += pd.Timedelta(days=1)
    return out


def _weekly_price_sigma(series: dict):
    """Std of the 7-day price change over the last VOLATILITY_DAYS days.

    This is the scale the uncertainty band is built from. It comes from the same
    price series the chart draws, so the band always reflects the volatility the
    user can actually see in the history. The window is three years: one year of
    heizoel24 prices swings too much to be a usable baseline.
    """
    prices = [
        float(v)
        for v in (series.get("price") or [])
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if len(prices) < 30:
        return None
    changes = (pd.Series(prices) - pd.Series(prices).shift(7)).dropna().tail(VOLATILITY_DAYS)
    if len(changes) < 8:
        return None
    sigma = float(changes.std())
    return sigma if sigma > 0 else None


def _prediction_of(payload: dict):
    """The price a /predict response carries, whichever model answered."""
    for key in ("prediction_next_week_price", "prediction"):
        value = (payload or {}).get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _forecast_curve(origin_date: str, origin_price: float, series: dict,
                    current_price=None, current_demand=None) -> dict:
    """Assemble the dotted forecast: weekly model first, forward model further out."""
    sigma = _weekly_price_sigma(series)
    points, errors = [], {}

    def band(horizon: int) -> float:
        if sigma is None:
            return 0.0
        return round(BAND_SIGMA * sigma * math.sqrt(max(int(horizon), 1) / 7.0), 2)

    def inputs(payload: dict) -> dict:
        body = {"date": payload["date"]}
        if current_price is not None:
            body["current_price"] = current_price
        if current_demand is not None:
            body["current_demand"] = current_demand
        return body

    # 1) weekly model: the week after the last known price date
    payload, status = _post_predict(inputs({"date": origin_date}))
    weekly_prediction = _prediction_of(payload) if status == 200 else None
    if weekly_prediction is None:
        errors["weekly"] = (payload or {}).get("error") or f"HTTP {status}"
        # 502 means the inference service itself is unreachable: asking it nine
        # more times would only multiply the waiting time for the user.
        if status == 502:
            return {"error": errors["weekly"], "errors": errors}
    else:
        points.append({
            "date": _forecast_week_midpoint(origin_date),
            "value": round(weekly_prediction, 2),
            "horizon_days": WEEKLY_HORIZON_DAYS,
            "model": "weekly",
            "origin_date": payload.get("origin_date") or origin_date,
            "band": band(WEEKLY_HORIZON_DAYS),
        })

    # 2) forward model: same origin date, further weeks ahead
    for horizon in FORWARD_HORIZON_STEPS:
        if horizon > FORECAST_MAX_HORIZON or horizon > FORWARD_MAX_SERVING_HORIZON:
            continue
        target = (pd.Timestamp(origin_date) + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")
        payload, status = _post_predict(inputs({"date": target}))
        prediction = _prediction_of(payload) if status == 200 else None
        if prediction is None:
            errors[f"forward_{horizon}d"] = (payload or {}).get("error") or f"HTTP {status}"
            if status == 502:
                errors["service"] = "prediction service unreachable; remaining horizons skipped"
                break
            continue
        points.append({
            "date": _forecast_week_midpoint(target),
            "value": round(prediction, 2),
            "horizon_days": payload.get("forecast_horizon_days") or horizon,
            "model": "forward",
            "origin_date": payload.get("origin_date") or origin_date,
            "band": band(horizon),
        })

    if not points:
        return {"error": errors.get("weekly") or "no forecast point available", "errors": errors}

    # The curve describes whole weeks ahead. The days of those weeks that are
    # already measured are sent along with it, so the chart can carry the real
    # price line through the forecast region instead of ending at the origin.
    last_forecast_date = max(p["date"] for p in points)

    return {
        "origin_date": origin_date,
        "origin_price": origin_price,
        "weekly_date": points[0]["date"] if points[0]["model"] == "weekly" else None,
        "max_horizon_days": max(p["horizon_days"] for p in points),
        "sigma_weekly": round(sigma, 2) if sigma else None,
        "points": points,
        "measured": _measured_prices(series, origin_date, last_forecast_date),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Rhine data: gauge snapshot + daily gauge series + monthly port cargo
# (written by src/collect_rhein_levels.py and src/collect_rhein_cargo.py).
# S3 first, bundled CSV fallback - same pattern as the price history above.
# ---------------------------------------------------------------------------
RHEIN_S3_BUCKET = os.environ.get("RHEIN_S3_BUCKET", "oilprice-features")
RHEIN_SNAPSHOT_KEY = "raw/rhein_snapshot.json"
RHEIN_LEVELS_KEY = "raw/rhein_levels.csv"
RHEIN_CARGO_KEY = "raw/rhein_cargo.csv"

# Low-water thresholds. Mirrors src/rhein_config.py - the frontend image only
# contains src/frontend (see Dockerfile.frontend), so the constants cannot be
# imported from the collectors there.
BASEL_LOW_WATER_CM = float(os.environ.get("BASEL_LOW_WATER_CM", 350.0))
KAUB_LOW_WATER_CM = float(os.environ.get("KAUB_LOW_WATER_CM", 30.0))

RHEIN_LEVEL_DAYS = int(os.environ.get("RHEIN_LEVEL_DAYS", 365))      # gauge chart window
RHEIN_CARGO_MONTHS = int(os.environ.get("RHEIN_CARGO_MONTHS", 36))  # cargo chart window
RHEIN_CACHE_TTL = int(os.environ.get("RHEIN_CACHE_TTL", 15 * 60))

# Cargo groups shown in the chart (NST-2007 codes; the legacy classification
# uses group 3 for mineral oil - see src/collect_rhein_cargo.py).
_OIL_GROUPS = ("07", "3")
_HEATING_OIL_GROUP = "07.23"
_TOTAL_GROUP = "TOTAL"

_rhein_cache = {"ts": 0.0, "data": None}


def _rhein_local_candidates(key: str) -> list:
    """Bundled (docker: /app/data/rhein/...) or repo-local fallback paths."""
    base = os.path.dirname(os.path.abspath(__file__))
    rel = key[len("raw/"):] if key.startswith("raw/") else key
    return [
        os.path.join(base, "data", "rhein", rel),  # docker image: /app/data/...
        os.path.abspath(os.path.join(base, "..", "..", "data", "rhein", rel)),  # local dev
    ]


def _read_rhein_object(key: str) -> tuple:
    """Read one Rhine file. Returns (payload, source) with source 's3' or 'local'."""
    try:
        s3 = boto3.client("s3")
        return s3.get_object(Bucket=RHEIN_S3_BUCKET, Key=key)["Body"].read(), "s3"
    except Exception as e:
        candidates = _rhein_local_candidates(key)
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            raise RuntimeError(f"s3://{RHEIN_S3_BUCKET}/{key} unavailable ({e}); no bundled file")
        print(f"warning: s3 read of {key} failed ({e}); using {path}")
        with open(path, "rb") as fh:
            return fh.read(), "local"


def _numbers(series) -> list:
    """JSON-safe list: NaN/None -> None, everything else float."""
    return [
        None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        for v in series
    ]


def _read_rhein_snapshot() -> tuple:
    """Current gauge snapshot (Basel + Kaub + bottleneck), realtime source."""
    body, source = _read_rhein_object(RHEIN_SNAPSHOT_KEY)
    return json.loads(body.decode("utf-8")), source


def _read_rhein_levels() -> tuple:
    """Daily gauge series for the chart (last RHEIN_LEVEL_DAYS days).

    Mirrors src/hw_rhein_feature_pipeline.py: the daily MINIMUM is the
    shipping-relevant value, and the bottleneck is min(Basel, Kaub) with a
    fallback to Basel while Kaub has no readings (WSV publishes only ~30 days).
    """
    body, source = _read_rhein_object(RHEIN_LEVELS_KEY)
    df = pd.read_csv(BytesIO(body))
    df["date"] = pd.to_datetime(df["date"])
    daily = (
        df.sort_values("date")
        .pivot_table(index="date", columns="station", values="level_cm_min", aggfunc="last")
        .sort_index()
    )
    for station in ("basel", "kaub"):
        if station not in daily.columns:
            daily[station] = np.nan

    daily["basel"] = daily["basel"].ffill(limit=2)  # a single missing day stays usable
    daily["kaub"] = daily["kaub"].ffill(limit=2)
    daily["bottleneck"] = daily[["basel", "kaub"]].min(axis=1)
    low = (daily["basel"] < BASEL_LOW_WATER_CM) | (daily["kaub"] < KAUB_LOW_WATER_CM)
    daily["low_water_flag"] = low.fillna(False).astype(int)

    daily = daily.tail(max(1, RHEIN_LEVEL_DAYS))
    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in daily.index],
        "basel": _numbers(daily["basel"]),
        "kaub": _numbers(daily["kaub"]),
        "bottleneck": _numbers(daily["bottleneck"]),
        "low_water_flag": [int(v) for v in daily["low_water_flag"]],
    }
    return series, source


def _read_rhein_cargo() -> tuple:
    """Monthly mineral-oil cargo of the Port of Switzerland (lagged data)."""
    body, source = _read_rhein_object(RHEIN_CARGO_KEY)
    df = pd.read_csv(BytesIO(body))
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["month"] + "-01"), freq="M")

    def series_for(groups, direction):
        selected = df[df["group"].isin(groups) & (df["direction"] == direction)]
        if selected.empty:
            return pd.Series(dtype="float64")
        # min(): duplicated rows of overlapping bulletins must agree
        return selected.groupby("month")["tonnes"].min()

    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    frame = pd.DataFrame(
        {
            "oil_zufuhr": series_for(_OIL_GROUPS, "zufuhr").reindex(months),
            "oil_abfuhr": series_for(_OIL_GROUPS, "abfuhr").reindex(months),
            "heating_oil_zufuhr": series_for((_HEATING_OIL_GROUP,), "zufuhr").reindex(months),
            "total_zufuhr": series_for((_TOTAL_GROUP,), "zufuhr").reindex(months),
        }
    )
    published = df.sort_values("published_date").groupby("month")["published_date"].last()
    frame["published_date"] = published.reindex(months)

    frame = frame.tail(max(1, RHEIN_CARGO_MONTHS))
    known = frame[frame["oil_zufuhr"].notna()]
    cargo = {
        "months": [p.strftime("%Y-%m") for p in frame.index],
        "oil_zufuhr": _numbers(frame["oil_zufuhr"]),
        "oil_abfuhr": _numbers(frame["oil_abfuhr"]),
        "heating_oil_zufuhr": _numbers(frame["heating_oil_zufuhr"]),
        "total_zufuhr": _numbers(frame["total_zufuhr"]),
        "latest_month": known.index[-1].strftime("%Y-%m") if len(known) else None,
        "latest_published": str(known["published_date"].iloc[-1]) if len(known) else None,
    }
    return cargo, source


def _read_rhein() -> dict:
    """Snapshot + gauge history + cargo history, with per-section error notes."""
    payload = {"thresholds": {"basel_cm": BASEL_LOW_WATER_CM, "kaub_cm": KAUB_LOW_WATER_CM}}
    errors = {}
    sources = []
    for name, reader in (
        ("snapshot", _read_rhein_snapshot),
        ("levels", _read_rhein_levels),
        ("cargo", _read_rhein_cargo),
    ):
        try:
            data, source = reader()
            payload[name] = data
            sources.append(source)
        except Exception as e:
            print(f"warning: rhein {name} unavailable: {e}")
            payload[name] = None
            errors[name] = str(e)
    payload["source"] = "s3" if sources and all(s == "s3" for s in sources) else "local"
    payload["errors"] = errors
    return payload


@app.route('/api/rhein')
def api_rhein():
    """Rhine gauge snapshot + gauge history + port cargo history (cached).

    Gauges are realtime; the cargo figures are published 4-7 weeks after the
    reporting month, so they are displayed as history only and are never used
    as an input to a live prediction.
    """
    now = time.time()
    if _rhein_cache["data"] is None or now - _rhein_cache["ts"] > RHEIN_CACHE_TTL:
        _rhein_cache["ts"] = now
        _rhein_cache["data"] = _read_rhein()
    return jsonify(_rhein_cache["data"])


@app.route('/api/forecast')
def api_forecast():
    """Dotted forecast curve for the price chart (cached per origin + inputs).

    Query params: current_price / current_demand (optional, the same values the
    forecast form sends). The curve always starts at the newest price date, so
    the chart can extend the history line into the future without the user
    having to press the button first. "measured" holds the real daily prices of
    every day from that origin date to the last forecast point (null where the
    source published nothing), which is what lets the measured line run on into
    the forecast region.
    """
    data, source = _price_cache["data"] if _price_cache["data"] else _read_price_history()
    origin = _last_price_point(data)
    if origin is None:
        return jsonify({"error": "no price history available"}), 503
    origin_date, origin_price = origin

    def number(name):
        raw = (request.args.get(name) or "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    current_price, current_demand = number("current_price"), number("current_demand")
    key = (origin_date, current_price, current_demand)
    now = time.time()
    cached = _forecast_cache
    if (cached["data"] is None or cached["key"] != key
            or now - cached["ts"] > FORECAST_CACHE_TTL):
        cached["ts"] = now
        cached["key"] = key
        cached["data"] = _forecast_curve(
            origin_date, origin_price, data, current_price, current_demand
        )
    payload = dict(cached["data"])
    payload["source"] = source
    return jsonify(payload)


@app.route('/api/history')
def api_history():
    """Price & demand history for the chart (cached, S3-first)."""
    now = time.time()
    if _price_cache["data"] is None or now - _price_cache["ts"] > PRICE_CACHE_TTL:
        _price_cache["ts"] = now
        _price_cache["data"] = _read_price_history()
    data, source = _price_cache["data"]
    return jsonify({"source": source, **data})

@app.route('/')
def home():
    # the label mapping is rendered into the page so the feature table can show
    # human-readable names instead of the raw feature-store columns
    return render_template('index.html', feature_labels=FEATURE_LABELS)

@app.route('/predict', methods=['POST'])
def predict():
    """Proxy to the inference service.

    Passes the backend response through (JSON + status code) so users see the
    backend's error messages (e.g. "no feature vector for date ...; latest
    available date: ...") instead of a raw HTTPError string.
    """
    data = request.get_json(silent=True) or {}
    payload, status = _post_predict(data)
    if not isinstance(payload, dict):
        payload = {"error": str(payload)}
    return jsonify(payload), status

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)