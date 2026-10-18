"""Fixture tests for the Kaub history archive (no network, no S3)."""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import collect_rhein_kaub_history as kaub  # noqa: E402
from rhein_config import KAUB  # noqa: E402


def _readings(dates, level=28.0, source="wsv"):
    return pd.DataFrame(
        {
            "timestamp_utc": [f"{d}T06:00:00+00:00" for d in dates],
            "station": KAUB["station"],
            "source": source,
            "level_cm": level,
            "discharge_m3s": 500.0,
        }
    )


def test_append_archive_is_append_only_and_deduped():
    old = _readings(["2026-09-01", "2026-09-02"])
    # second window overlaps the first one and corrects the 2nd reading
    fresh = _readings(["2026-09-02", "2026-09-03"], level=31.0)

    archive = kaub.append_archive(old, fresh)
    assert len(archive) == 3
    assert archive["timestamp_utc"].tolist() == sorted(archive["timestamp_utc"].tolist())
    corrected = archive[archive["timestamp_utc"] == "2026-09-02T06:00:00+00:00"].iloc[0]
    assert corrected["level_cm"] == 31.0  # the fresh (corrected) value wins


def test_append_archive_handles_empty_inputs():
    archive = kaub.append_archive(None, _readings(["2026-09-01"]))
    assert len(archive) == 1
    empty = kaub.append_archive(pd.DataFrame(), pd.DataFrame())
    assert empty.empty and list(empty.columns) == kaub.RAW_COLUMNS


def test_refresh_daily_levels_aggregates_the_archive(monkeypatch):
    archive = _readings(["2026-09-01", "2026-09-02", "2026-09-02"], level=30.0)
    archive.loc[archive.index[-1], "level_cm"] = 25.0

    saved = {}
    monkeypatch.setattr("collect_rhein_levels.load_csv", lambda key, local: None)
    monkeypatch.setattr(
        "collect_rhein_levels.save_csv",
        lambda df, key, local: saved.update({"df": df}),
    )

    daily = kaub.refresh_daily_levels(archive)
    assert list(daily["date"]) == ["2026-09-01", "2026-09-02"]
    day = daily[daily["date"] == "2026-09-02"].iloc[0]
    assert day["level_cm_min"] == 25.0
    assert day["level_cm_max"] == 30.0
    assert day["station"] == KAUB["station"]
    assert len(saved["df"]) == 2
