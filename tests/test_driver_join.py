"""Fixture tests for the Rhine/market driver join (no network, no Hopsworks)."""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import hw_config  # noqa: E402
import hw_feature_pipeline as weekly  # noqa: E402
import hw_forward_feature_pipeline as forward  # noqa: E402


def _dataset():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"date": dates})
    for col in weekly.FEATURES:
        frame[col] = np.arange(len(dates), dtype="float64")
    for col in weekly.LABELS:
        frame[col] = np.arange(len(dates), dtype="float64")
    return frame


def _drivers():
    # only 4 of the 10 days have driver rows -> the join must keep all 10 rows
    dates = pd.date_range("2026-01-05", periods=4, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "basel_level_cm": [400.0, 380.0, 360.0, 340.0],
            "basel_level_min_7d": [380.0, 360.0, 340.0, 340.0],
            "basel_level_zscore_30d": [-1.0, -1.2, -1.5, -1.8],
            "basel_low_days_30d": [2.0, 3.0, 4.0, 5.0],
            "basel_level_trend_7d": [-10.0, -12.0, -15.0, -18.0],
            "rhein_low_water_flag": [0, 0, 1, 1],
            "market_price_chf_100l": [60.0, 61.0, 62.0, 63.0],
            "market_price_chf_100l_mean_7d": [59.0, 60.0, 61.0, 62.0],
            "market_price_chf_100l_chg_7d_pct": [1.0, 1.5, 2.0, 2.5],
            "gasoil_usd_tonne": [800.0, 810.0, 820.0, 830.0],
            "usd_chf": [0.82, 0.82, 0.83, 0.83],
        }
    )


def test_weekly_feature_list_contains_the_drivers():
    features = hw_config.weekly_model_features()
    assert features[: len(weekly.FEATURES)] == weekly.FEATURES
    for col in hw_config.RHEIN_MODEL_FEATURES + hw_config.MARKET_MODEL_FEATURES:
        assert col in features
    # the label must never be part of the feature list
    assert hw_config.TARGET not in features


def test_forward_feature_list_uses_origin_prefix():
    features = hw_config.forward_model_features()
    assert "origin_basel_level_cm" in features
    assert "origin_market_price_chf_100l" in features
    assert "forecast_horizon_days" in features


def test_attach_drivers_keeps_every_price_row():
    out = weekly.attach_drivers(_dataset(), _drivers(), hw_config.RHEIN_MODEL_FEATURES + hw_config.MARKET_MODEL_FEATURES)
    assert len(out) == 10
    # 'date' is the primary key of the feature group and must survive the join
    assert out.columns.tolist()[0] == "date"
    assert out["basel_level_cm"].notna().sum() == 4
    assert out["market_price_chf_100l"].notna().sum() == 4
    # drivers are appended after the existing columns, labels stay in place
    assert out.columns.tolist()[-5:] == hw_config.MARKET_MODEL_FEATURES
    assert "target_next_week_price" in out.columns


def test_read_driver_csv_fills_missing_file_with_nulls(tmp_path):
    missing = tmp_path / "does-not-exist.csv"
    frame = weekly._read_driver_csv(missing, ["basel_level_cm"], "Rhine gauge")
    assert list(frame.columns) == ["date", "basel_level_cm"]
    assert frame["basel_level_cm"].isna().all()


def test_read_driver_csv_fills_missing_column(tmp_path):
    path = tmp_path / "drivers.csv"
    pd.DataFrame({"date": ["2026-01-01"], "basel_level_cm": [400.0]}).to_csv(path, index=False)
    frame = weekly._read_driver_csv(path, ["basel_level_cm", "rhein_low_water_flag"], "Rhine gauge")
    assert frame["rhein_low_water_flag"].isna().all()
    assert frame["basel_level_cm"].iloc[0] == 400.0


def test_forward_driver_columns_carry_weekend_gap_only(monkeypatch):
    daily = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=12, freq="D")})
    drivers = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "market_price_chf_100l": [60.0] * 6,
        }
    )
    monkeypatch.setattr(forward, "_driver_frames", lambda: (drivers, ["market_price_chf_100l"]))

    out, columns = forward._attach_driver_columns(daily)
    assert columns == ["market_price_chf_100l"]
    # days 7-9 (index 6..8) are within the limit=3 carry, day 10 is not
    assert out["market_price_chf_100l"].iloc[7] == 60.0
    assert np.isnan(out["market_price_chf_100l"].iloc[9])


def test_forward_origin_features_prefix_the_drivers(monkeypatch):
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "price": [100.0] * 40,
            "demand": [10.0] * 40,
            "basel_level_cm": [400.0] * 40,
        }
    )
    origin = forward._origin_features(daily, driver_cols=["basel_level_cm"])
    assert origin["origin_basel_level_cm"].notna().all()
    assert origin["origin_price_mean_7d"].notna().all()
