"""Fixture tests for the oil-market feature pipeline (no network)."""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import collect_oil_market_prices as market  # noqa: E402
import hw_oil_market_feature_pipeline as pipeline  # noqa: E402
from market_config import (  # noqa: E402
    HEIZOEL_100L_UNITS_PER_TONNE,
    MARKET_FEATURES,
    MARKET_RAW_COLUMNS,
)


def _raw():
    """10 trading days: 800 USD/t gasoil, 0.90 USD/CHF."""
    dates = pd.bdate_range("2026-01-05", periods=10)
    prices = [800.0] * 5 + [820.0] * 5
    return pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "gasoil_usd_tonne": prices,
            "usd_chf": [0.90] * 10,
            "gasoil_chf_tonne": [p * 0.90 for p in prices],
            "market_price_chf_100l": [p * 0.90 / HEIZOEL_100L_UNITS_PER_TONNE for p in prices],
            "gasoil_source": ["yahoo:LGO=F"] * 10,
            "fx_source": ["yahoo:USDCHF=X"] * 10,
        }
    )


def test_chf_100l_conversion_matches_the_tonne_density():
    # 1 t Heizoel extra leicht = 1000 kg / 0.845 kg/l = 1183.43 l = 11.8343 x 100 l
    assert round(HEIZOEL_100L_UNITS_PER_TONNE, 4) == 11.8343
    frame = pipeline.build_market_features(_raw())
    expected = 800.0 * 0.90 / HEIZOEL_100L_UNITS_PER_TONNE
    assert frame["market_price_chf_100l"].iloc[0] == pytest.approx(expected, abs=1e-4)


def test_market_features_columns_and_windows():
    frame = pipeline.build_market_features(_raw())
    assert list(frame.columns) == ["date"] + MARKET_FEATURES + ["gasoil_source", "fx_source"]
    assert len(frame) == 10
    # 7-day mean of the first five (identical) values equals the level itself
    assert frame["market_price_chf_100l_mean_7d"].iloc[4] == pytest.approx(
        frame["market_price_chf_100l"].iloc[4], abs=1e-4
    )
    # the 7-day change compares day t with t-7 (trading days)
    level = frame["market_price_chf_100l"]
    expected_chg = (level.iloc[9] - level.iloc[2]) / level.iloc[2] * 100
    assert frame["market_price_chf_100l_chg_7d_pct"].iloc[9] == pytest.approx(expected_chg, abs=1e-3)
    assert expected_chg == pytest.approx((820.0 - 800.0) / 800.0 * 100, abs=1e-3)


def test_build_market_frame_output_columns():
    gasoil = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                           "gasoil_usd_tonne": [800.0, 810.0], "gasoil_source": ["a", "a"]})
    fx = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                       "usd_chf": [0.9, 0.9], "fx_source": ["b", "b"]})

    original_gasoil = market.fetch_gasoil_usd_tonne
    original_fx = market.fetch_usd_chf
    market.fetch_gasoil_usd_tonne = lambda period: gasoil
    market.fetch_usd_chf = lambda period: fx
    try:
        frame = market.build_market_frame("2y")
    finally:
        market.fetch_gasoil_usd_tonne = original_gasoil
        market.fetch_usd_chf = original_fx

    assert list(frame.columns) == MARKET_RAW_COLUMNS
    assert frame["gasoil_chf_tonne"].iloc[0] == pytest.approx(720.0, abs=1e-4)
    assert frame["market_price_chf_100l"].iloc[0] == pytest.approx(
        720.0 / HEIZOEL_100L_UNITS_PER_TONNE, abs=1e-4
    )


def test_parse_ecb_daily_cross_rate():
    xml = """
    <Cube><Cube time='2026-10-16'>
      <Cube currency='USD' rate='1.1253'/>
      <Cube currency='CHF' rate='0.9272'/>
      <Cube currency='GBP' rate='0.7'/>
    </Cube></Cube>
    """
    frame = market._parse_ecb_daily(xml)
    assert len(frame) == 1
    # USD/CHF = EURCHF / EURUSD
    assert frame["usd_chf"].iloc[0] == pytest.approx(0.9272 / 1.1253)


def test_merge_raw_upserts_by_date():
    existing = _raw().iloc[:8]
    fresh = _raw().iloc[6:]
    fresh.loc[fresh.index[-1], "gasoil_usd_tonne"] = 999.0

    merged = market.merge_raw(existing, fresh)
    assert len(merged) == 10
    assert merged["date"].tolist() == sorted(merged["date"].tolist())
    last = merged[merged["date"] == fresh["date"].iloc[-1]].iloc[0]
    assert last["gasoil_usd_tonne"] == 999.0
