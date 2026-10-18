"""Fixture tests for the Rhine gauge collector (no network)."""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import collect_rhein_levels as levels  # noqa: E402
from rhein_config import BASEL, KAUB  # noqa: E402


def _raw(rows):
    """rows: (timestamp, station, source, level_cm, discharge_m3s)"""
    frame = pd.DataFrame(
        rows, columns=["timestamp", "station", "source", "level_cm", "discharge_m3s"]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame


def test_aggregate_daily_uses_utc_calendar_days():
    # WSV publishes local time (+02:00). 2026-10-17 01:30 CEST is 2026-10-16
    # 23:30 UTC, so it must land in the 16th, not the 17th.
    raw = _raw(
        [
            ("2026-10-17T01:30:00+02:00", "basel", "wsv", 480.0, None),
            ("2026-10-17T02:30:00+02:00", "basel", "wsv", 481.0, None),
            ("2026-10-17T00:00:00+00:00", "basel", "data.bs.ch", 482.0, 390.0),
            ("2026-10-17T23:00:00+00:00", "basel", "data.bs.ch", 486.0, 394.0),
        ]
    )
    daily = levels.aggregate_daily(raw)

    assert list(daily["date"]) == ["2026-10-16", "2026-10-17"]
    day = daily[daily["date"] == "2026-10-17"].iloc[0]
    assert day["level_cm_min"] == 481.0
    assert day["level_cm_max"] == 486.0
    assert day["level_cm_last"] == 486.0
    assert day["n_level_obs"] == 3
    assert day["sources"] == "data.bs.ch|wsv"
    assert day["discharge_m3s_mean"] == 392.0


def test_aggregate_daily_separates_stations():
    raw = _raw(
        [
            ("2026-10-17T10:00:00+00:00", "basel", "data.bs.ch", 480.0, 390.0),
            ("2026-10-17T10:15:00+00:00", "kaub", "wsv", 12.0, 470.0),
            ("2026-10-17T10:30:00+00:00", "kaub", "wsv", 10.0, 472.0),
        ]
    )
    daily = levels.aggregate_daily(raw).set_index("station")

    assert daily.loc["basel", "level_cm_min"].item() == 480.0
    assert daily.loc["kaub", "level_cm_min"].item() == 10.0
    assert daily.loc["kaub", "level_cm_max"].item() == 12.0
    assert daily.loc["kaub", "n_level_obs"].item() == 2


def test_aggregate_daily_empty():
    daily = levels.aggregate_daily(pd.DataFrame())
    assert list(daily.columns) == levels.LEVEL_COLUMNS
    assert daily.empty


def test_merge_levels_overwrites_and_reports_corrections(capsys):
    existing = pd.DataFrame(
        [
            {
                "date": "2026-10-17",
                "station": "basel",
                "level_cm_last": 486.0,
                "level_cm_mean": 482.0,
                "level_cm_min": 480.0,
                "level_cm_max": 486.0,
                "discharge_m3s_mean": 390.0,
                "n_level_obs": 10,
                "sources": "data.bs.ch",
            }
        ]
    )
    fresh = pd.DataFrame(
        [
            {
                "date": "2026-10-17",
                "station": "basel",
                "level_cm_last": 486.0,
                "level_cm_mean": 483.0,
                "level_cm_min": 479.0,  # corrected by the gauge operator
                "level_cm_max": 486.0,
                "discharge_m3s_mean": 391.0,
                "n_level_obs": 12,
                "sources": "data.bs.ch|wsv",
            },
            {
                "date": "2026-10-18",
                "station": "basel",
                "level_cm_last": 484.0,
                "level_cm_mean": 482.0,
                "level_cm_min": 480.0,
                "level_cm_max": 486.0,
                "discharge_m3s_mean": 388.0,
                "n_level_obs": 12,
                "sources": "data.bs.ch",
            },
        ]
    )

    merged = levels.merge_levels(existing, fresh)

    assert len(merged) == 2
    row = merged[merged["date"] == "2026-10-17"].iloc[0]
    assert row["level_cm_min"] == 479.0  # fresh value wins
    assert row["n_level_obs"] == 12
    assert "corrected 2026-10-17 basel" in capsys.readouterr().out


def test_low_water_thresholds_are_station_specific():
    assert BASEL["low_water_cm"] == 350.0
    assert KAUB["low_water_cm"] == 30.0
    # Kaub is downstream and shallow: its zero point is not comparable to Basel
    assert KAUB["bs_ch_dataset"] is None
    assert BASEL["wsv_discharge"] is False
