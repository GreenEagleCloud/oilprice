"""Shared configuration for the oil-market drivers (ICE Gasoil + USD/CHF).

Used by:
    * src/collect_oil_market_prices.py        (raw market data -> S3)
    * src/hw_oil_market_feature_pipeline.py   (Hopsworks feature group/view)
    * src/hw_config.py                        (join into the training views)

Why these two series
--------------------
The purchase price of heating oil in Switzerland is set by the international
product price (ICE Gasoil, quoted in USD per tonne) converted into CHF. Two
independent drivers therefore have to be modelled:

    gasoil_usd_tonne  ICE Gasoil (Light Diesel) close, USD/tonne
    usd_chf           USD/CHF spot close

    gasoil_chf_tonne        = gasoil_usd_tonne * usd_chf
    market_price_chf_100l   = gasoil_chf_tonne / HEIZOEL_100L_UNITS_PER_TONNE

`market_price_chf_100l` is the landed-cost reference in the same unit the
heizoel24 target uses (CHF per 100 litres), so the model sees the driver in the
target's scale without any unit mismatch.

Units / constants
-----------------
Heizöl extra leicht density 0.845 kg/l:
    1 tonne = 1000 kg / 0.845 kg/l = 1183.43 l = 11.8343 units of 100 l
    1 tonne = 1183.43 l / 158.987 l/bbl = 7.4433 barrels

Data sources (verified live 2026-10-18)
--------------------------------------
1. Yahoo Finance via yfinance (primary)
   USDCHF=X  USD/CHF spot close, daily, since 2003-09-17 (verified).
   LGO=F     ICE Gasoil 100 t futures. Yahoo no longer serves this symbol
             (quoteSummary 404 "may be delisted", 0 rows for every period),
             so the pipeline falls back to the Brent proxy below. Override the
             symbol with OILPRICE_GASOIL_SYMBOL once a licensed ICE feed exists.

2. FRED (fallback for the product price, no API key needed for fredgraph.csv)
   DCOILBRENTEU Brent crude spot, USD/barrel, daily since 1987 -> converted to
                USD/tonne with BARREL_PER_TONNE and flagged as "brent_proxy".
   A Brent proxy is a close but not identical driver of gasoil; the source flag
   is stored in the feature group so the substitution is auditable.

3. ECB euro reference rates (fallback for the currency)
   eurofxref-hist.csv / eurofxref-daily.xml: EUR per USD and EUR per CHF, so
   USD/CHF = EURCHF / EURUSD. FRED has no USD/CHF series (DEXCHUS is CNY/USD).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Unit conversion (Heizöl extra leicht, 0.845 kg/l)
# ---------------------------------------------------------------------------
GASOIL_DENSITY_KG_L = 0.845
LITER_PER_TONNE = 1000.0 / GASOIL_DENSITY_KG_L              # 1183.43 l/t
HEIZOEL_100L_UNITS_PER_TONNE = LITER_PER_TONNE / 100.0      # 11.8343 x 100 l/t
LITER_PER_BARREL = 158.987                                  # 1 bbl = 158.987 l
BARREL_PER_TONNE = LITER_PER_TONNE / LITER_PER_BARREL       # 7.4433 bbl/t

# ---------------------------------------------------------------------------
# Symbols / series
# ---------------------------------------------------------------------------
# Yahoo no longer serves ICE Gasoil (LGO=F is delisted there), so the Yahoo
# gasoil request is opt-in: set OILPRICE_GASOIL_SYMBOL to a Yahoo futures
# symbol (e.g. a licensed feed or a restored Yahoo symbol) to use it. Empty
# default -> go straight to the Brent proxy, which avoids a guaranteed 404 on
# every run (that 404 makes Yahoo rate-limit the follow-up requests too).
GASOIL_SYMBOL = os.getenv("OILPRICE_GASOIL_SYMBOL", "")
FX_SYMBOL = os.getenv("OILPRICE_FX_SYMBOL", "USDCHF=X")
BRENT_SYMBOL = os.getenv("OILPRICE_BRENT_SYMBOL", "BZ=F")

# FRED (no API key needed for fredgraph.csv):
#   DCOILBRENTEU  Brent crude spot, USD per barrel, daily since 1987
# There is no FRED series for USD/CHF (DEXCHUS is CNY/USD), so the FX fallback
# uses the ECB euro reference rates instead.
FRED_BRENT_SERIES = "DCOILBRENTEU"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# ECB euro reference rates: EUR per USD and EUR per CHF -> USD/CHF = CHF/USD.
# eurofxref-hist.csv is the daily history since 1999; eurofxref-daily.xml is
# the latest published day (the history file sometimes lags a day).
ECB_HIST_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
ECB_DAILY_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"

# Default history window of the collector. "max" is deliberately NOT the
# default: Yahoo answers large requests with an empty frame (rate limit), while
# a 10y window is served reliably. merge_raw() upserts by date, so the stored
# history grows with every daily run and the ECB fallback already provides the
# full FX history since 1999.
DEFAULT_PERIOD = os.getenv("OILPRICE_MARKET_PERIOD", "10y")

# ---------------------------------------------------------------------------
# Storage (same bucket as the other collectors)
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("RHEIN_S3_BUCKET", "oilprice-features")
MARKET_RAW_KEY = "raw/oil_market_prices.csv"

LOCAL_DIR = REPO_ROOT / "data" / "oil"
LOCAL_MARKET_CSV = LOCAL_DIR / "oil_market_prices.csv"
FEATURES_LOCAL_CSV = REPO_ROOT / "data" / "features" / "oil_market_features.csv"

# ---------------------------------------------------------------------------
# Feature group / view (Hopsworks)
# ---------------------------------------------------------------------------
MARKET_FEATURE_GROUP = "oil_market"
MARKET_FEATURE_GROUP_VERSION = 1
MARKET_FEATURE_VIEW = "oil_market_daily"
MARKET_FEATURE_VIEW_VERSION = 1

# Raw driver columns (written by the collector) + the smoothed variants that
# the model actually uses.
MARKET_RAW_COLUMNS = [
    "date",
    "gasoil_usd_tonne",
    "usd_chf",
    "gasoil_chf_tonne",
    "market_price_chf_100l",
    "gasoil_source",
    "fx_source",
]

MARKET_FEATURES = [
    "gasoil_usd_tonne",
    "usd_chf",
    "gasoil_chf_tonne",
    "market_price_chf_100l",
    "market_price_chf_100l_mean_7d",
    "market_price_chf_100l_chg_7d_pct",
]

# Source flags are stored in the feature group for auditability but are not
# model features (they are constant while a fallback source is in use).
MARKET_SOURCE_COLUMNS = ["gasoil_source", "fx_source"]
