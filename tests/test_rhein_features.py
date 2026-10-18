"""Fixture tests for the Rhine feature pipeline (no network, no Hopsworks)."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import hw_rhein_feature_pipeline as pipeline  # noqa: E402
from rhein_config import RHEIN_CARGO_FEATURES, RHEIN_LEVEL_FEATURES  # noqa: E402


def _levels():
    """40 days of Basel (400 cm, then a drop to 300 cm) + 20 days of Kaub."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    basel = [400.0] * 30 + [300.0] * 10
    rows = [
        {"date": day.date().isoformat(), "station": "basel", "level_cm_min": level,
         "level_cm_mean": level, "level_cm_max": level, "discharge_m3s_mean": 400.0}
        for day, level in zip(dates, basel)
    ]
    kaub_dates = dates[20:]
    rows += [
        {"date": day.date().isoformat(), "station": "kaub", "level_cm_min": 20.0,
         "level_cm_mean": 21.0, "level_cm_max": 22.0, "discharge_m3s_mean": 500.0}
        for day in kaub_dates
    ]
    return pd.DataFrame(rows)


def test_level_features_shape_and_columns():
    features = pipeline.build_level_features(_levels())
    assert list(features.columns) == ["date"] + RHEIN_LEVEL_FEATURES
    assert len(features) == 40
    assert features["date"].tolist()[0] == "2026-01-01"
    # Kaub is only published for the last ~30 days by WSV
    assert features["kaub_level_cm"].notna().sum() == 20


def test_bottleneck_is_the_lower_gauge():
    features = pipeline.build_level_features(_levels())
    known = features[features["kaub_level_cm"].notna()]
    # Kaub (20 cm) is far below Basel (300-400 cm) -> Kaub is the bottleneck
    assert (known["bottleneck_level_cm"] == known["kaub_level_cm"]).all()
    # without Kaub the bottleneck falls back to Basel (upper bound)
    missing = features[features["kaub_level_cm"].isna()]
    assert (missing["bottleneck_level_cm"] == missing["basel_level_cm"]).all()


def test_low_water_flag_and_low_days():
    features = pipeline.build_level_features(_levels())
    first_kaub_day = features[features["kaub_level_cm"].notna()].index[0]

    # before the first Kaub reading nothing is known about the bottleneck
    assert features["rhein_low_water_flag"].iloc[0] == 0
    # Kaub at 20 cm is below its 30 cm threshold from the first day on
    assert features["rhein_low_water_flag"].iloc[first_kaub_day] == 1
    assert features["kaub_low_days_30d"].iloc[first_kaub_day + 19] == 20
    # Basel (400 cm) only drops below its 350 cm threshold in the last 10 days
    assert features["basel_low_days_30d"].iloc[-1] == 10


def test_rolling_windows_and_zscore():
    features = pipeline.build_level_features(_levels())
    # rolling windows need min_periods observations
    assert np.isnan(features["basel_level_min_7d"].iloc[2])
    assert features["basel_level_min_7d"].iloc[3] == pytest.approx(400.0)
    assert features["basel_level_mean_7d"].iloc[36] == pytest.approx(300.0)
    # index 33 still mixes the 400 cm and the 300 cm days
    assert features["basel_level_mean_7d"].iloc[33] == pytest.approx(342.86, abs=0.01)
    # the 7-day trend is negative while the level drops, zero while it is flat
    assert features["basel_level_trend_7d"].iloc[35] < 0
    assert features["basel_level_trend_7d"].iloc[10] == pytest.approx(0.0)
    # z-score needs 15 observations
    assert np.isnan(features["basel_level_zscore_30d"].iloc[10])
    assert features["basel_level_zscore_30d"].iloc[35] < 0


def test_missing_basel_rows_are_dropped():
    frame = _levels()
    frame = frame[frame["date"] != "2026-01-05"]
    features = pipeline.build_level_features(frame)
    assert "2026-01-05" in set(features["date"])  # single gap is forward-filled
    assert features["basel_level_cm"].notna().all()


def _cargo():
    rows = []
    for month, value in [("2025-01", 100.0), ("2025-02", 200.0), ("2025-03", 150.0),
                         ("2026-01", 150.0), ("2026-02", 300.0)]:
        rows.append({"month": month, "group": "07", "direction": "zufuhr",
                     "tonnes": value, "published_date": f"{month}-15",
                     "current_ytd_tonnes": value * 2})
        rows.append({"month": month, "group": "07.23", "direction": "zufuhr",
                     "tonnes": value * 0.6, "published_date": f"{month}-15",
                     "current_ytd_tonnes": np.nan})
    return pd.DataFrame(rows)


def test_cargo_features_expand_the_month_calendar():
    features = pipeline.build_cargo_features(_cargo())
    months = list(features["month"])
    assert months[0] == "2025-01" and months[-1] == "2026-02"
    # months without a bulletin are kept as gaps so MoM/YoY stay calendar-correct
    assert "2025-04" in months and "2025-12" in months
    assert features[features["month"] == "2025-04"]["oil_zufuhr_t"].isna().all()
    assert set(RHEIN_CARGO_FEATURES + ["month", "published_date"]) >= set(features.columns)


def test_cargo_mom_and_yoy_use_the_right_months():
    features = pipeline.build_cargo_features(_cargo()).set_index("month")
    assert features.loc["2025-02", "oil_zufuhr_mom_pct"] == pytest.approx(100.0)
    assert features.loc["2025-03", "oil_zufuhr_mom_pct"] == pytest.approx(-25.0)
    # 2026-02 (300) vs 2025-02 (200)
    assert features.loc["2026-02", "oil_zufuhr_yoy_pct"] == pytest.approx(50.0)
    assert features.loc["2026-02", "oil_zufuhr_prev_month_t"] == pytest.approx(150.0)
    assert features.loc["2026-02", "gas_diesel_heizoel_zufuhr_t"] == pytest.approx(180.0)
    assert features.loc["2026-02", "oil_zufuhr_ytd_t"] == pytest.approx(600.0)
