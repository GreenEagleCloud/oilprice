"""Fixture tests for the frontend Rhine endpoint (no network, no S3).

The endpoint reads the bundled files in data/rhein/ (the S3 fallback of
src/frontend/app.py), so the tests also verify the fallback path works.
"""

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "frontend"))

import app as frontend  # noqa: E402  (src/frontend/app.py)
from rhein_config import BASEL, KAUB  # noqa: E402


@pytest.fixture(scope="module")
def payload():
    """Fresh /api/rhein payload, read once per module (bundled files)."""
    frontend._rhein_cache["data"] = None
    response = frontend.app.test_client().get("/api/rhein")
    assert response.status_code == 200
    return response.get_json()


def test_endpoint_returns_all_sections(payload):
    assert set(payload) >= {"snapshot", "levels", "cargo", "thresholds", "source", "errors"}
    assert payload["errors"] == {}
    assert payload["source"] in ("s3", "local")
    assert payload["snapshot"] and payload["levels"] and payload["cargo"]


def test_thresholds_match_rhein_config(payload):
    # the frontend repeats the thresholds because its image has no src/rhein_config.py
    assert payload["thresholds"]["basel_cm"] == BASEL["low_water_cm"]
    assert payload["thresholds"]["kaub_cm"] == KAUB["low_water_cm"]


def test_snapshot_bottleneck_is_the_lower_station(payload):
    snapshot = payload["snapshot"]
    stations = snapshot["stations"]
    levels = [s["wsv_level_cm"] for s in stations.values() if s.get("wsv_level_cm") is not None]
    assert snapshot["bottleneck_level_cm"] == min(levels)
    assert snapshot["bottleneck_station"] in stations
    expected_flag = 1 if any(
        s["wsv_level_cm"] < s["low_water_cm"]
        for s in stations.values()
        if s.get("wsv_level_cm") is not None
    ) else 0
    assert snapshot["low_water_flag"] == expected_flag


def test_levels_series_is_aligned(payload):
    levels = payload["levels"]
    dates = levels["dates"]
    assert dates == sorted(dates)
    assert len(dates) == len(levels["basel"]) == len(levels["kaub"]) == len(levels["bottleneck"])
    assert len(dates) == len(levels["low_water_flag"])
    assert all(len(d) == 10 for d in dates)  # ISO yyyy-mm-dd

    # bottleneck = min(Basel, Kaub); without Kaub readings it falls back to Basel
    for basel, kaub, bottleneck in zip(levels["basel"], levels["kaub"], levels["bottleneck"]):
        if kaub is None:
            assert bottleneck == basel
        else:
            assert bottleneck == min(basel, kaub)

    # Kaub is published for the last ~30 days only by WSV -> readings must be
    # contiguous up to the last day (no holes in the middle of the window)
    known = [i for i, v in enumerate(levels["kaub"]) if v is not None]
    assert known
    assert known == list(range(known[0], len(levels["kaub"])))


def test_low_water_flag_of_the_series(payload):
    levels = payload["levels"]
    thresholds = payload["thresholds"]
    for basel, kaub, flag in zip(levels["basel"], levels["kaub"], levels["low_water_flag"]):
        low_basel = basel is not None and basel < thresholds["basel_cm"]
        low_kaub = kaub is not None and kaub < thresholds["kaub_cm"]
        assert flag == int(low_basel or low_kaub)


def test_cargo_series_is_monthly_and_lagged(payload):
    cargo = payload["cargo"]
    months = cargo["months"]
    assert months == sorted(months)
    assert len(months) == len(cargo["oil_zufuhr"]) == len(cargo["heating_oil_zufuhr"])
    assert cargo["latest_month"] == max(m for m, v in zip(months, cargo["oil_zufuhr"]) if v is not None)

    # bulletins are published 4-7 weeks after the reporting month
    latest = cargo["latest_month"]
    published = cargo["latest_published"]
    assert published[:7] >= latest
    assert published > latest


def test_index_page_contains_the_rhein_card():
    html = frontend.app.test_client().get("/").get_data(as_text=True)
    for marker in ("/api/rhein", "rheinChart", "cargoChart", "rheinStats", "Engstelle"):
        assert marker in html
