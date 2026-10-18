"""Tests for /api/forecast - the dotted forecast curve of the price chart.

The inference service is never contacted: _post_predict is replaced by a fake
that answers every date, so these tests cover how the frontend assembles the
curve - weekly model for the first week, forward model for the weeks after it,
the volatility band, the horizon limit, the partial-failure behaviour and the
caching.
"""

import math
import pathlib
import sys

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "frontend"))

import app as frontend  # noqa: E402

ORIGIN = "2026-10-09"


def price_series(days: int = 140) -> dict:
    """A price series whose newest day is the forecast origin date."""
    dates = pd.date_range(end=ORIGIN, periods=days, freq="D")
    price = [100 + 4 * math.sin(i / 6) + 0.5 * (i % 7) for i in range(days)]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "price": price,
        "demand": [100.0] * days,
    }


def fake_response(payload):
    """The answer a healthy inference service would give for this request."""
    horizon = (pd.Timestamp(payload["date"]) - pd.Timestamp(ORIGIN)).days
    if horizon <= 0:
        return {
            "date": payload["date"],
            "model": "weekly",
            "prediction_next_week_price": 150.0,
        }, 200
    return {
        "date": payload["date"],
        "origin_date": ORIGIN,
        "forecast_horizon_days": horizon,
        "model": "forward_forecast",
        "prediction_next_week_price": 150.0 + 0.25 * horizon,
    }, 200


@pytest.fixture()
def fake_backend(monkeypatch):
    """Stand-in for the inference service; records the requests it received."""
    calls = []

    def recorded(payload):
        calls.append(dict(payload))
        return fake_response(payload)

    monkeypatch.setattr(frontend, "_post_predict", recorded)
    monkeypatch.setitem(frontend._price_cache, "data", (price_series(), "local"))
    monkeypatch.setitem(frontend._forecast_cache, "data", None)
    monkeypatch.setitem(frontend._forecast_cache, "key", None)
    return calls


def get_forecast(query=""):
    with frontend.app.test_client() as client:
        response = client.get(f"/api/forecast{query}")
        return response.status_code, response.get_json()


def test_curve_starts_with_the_weekly_model(fake_backend):
    status, body = get_forecast()
    assert status == 200
    assert body["origin_date"] == ORIGIN
    first = body["points"][0]
    assert first["model"] == "weekly"
    assert first["horizon_days"] == 7
    assert first["value"] == 150.0
    # The weekly model answers for the week AFTER the origin date, so the point
    # belongs to the middle of that week - not to the origin date itself.
    assert first["date"] == "2026-10-13"


def test_forward_model_extends_the_curve_to_three_months(fake_backend):
    """The cone on the chart is drawn out to three months, which is the longest
    horizon the forward model serves (weekly steps end at 84 days)."""
    _, body = get_forecast()
    forward = [p for p in body["points"] if p["model"] == "forward"]
    horizons = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
    assert [p["horizon_days"] for p in forward] == horizons
    assert body["max_horizon_days"] == 84
    expected = [
        (pd.Timestamp(ORIGIN) + pd.Timedelta(days=h + 4)).strftime("%Y-%m-%d")
        for h in horizons
    ]
    assert [p["date"] for p in forward] == expected
    # the weekly point and the first forward point describe different weeks, so
    # they must not collapse onto the same x position
    assert body["points"][0]["date"] != forward[0]["date"]


def test_sigma_measures_the_seven_day_change_not_the_daily_noise():
    """A sawtooth that is flat for six days and jumps by 20 CHF on the seventh
    has no daily noise but a large weekly move - the band must see the weekly
    move, otherwise it is far too narrow to mean anything."""
    prices = [100.0] * 7 + [120.0] * 7 + [100.0] * 7 + [120.0] * 7 + [100.0] * 7
    series = {"dates": [f"2026-01-{d:02d}" for d in range(1, 36)], "price": prices}
    sigma = frontend._weekly_price_sigma(series)
    assert sigma == pytest.approx(20.0, abs=0.6)

    flat = {"dates": [f"2026-01-{d:02d}" for d in range(1, 36)], "price": [100.0] * 35}
    assert frontend._weekly_price_sigma(flat) is None  # no volatility, no band


def test_band_grows_with_the_square_root_of_the_horizon(fake_backend):
    _, body = get_forecast()
    sigma = body["sigma_weekly"]
    assert sigma and sigma > 0
    for point in body["points"]:
        expected = round(frontend.BAND_SIGMA * sigma * math.sqrt(point["horizon_days"] / 7), 2)
        assert point["band"] == pytest.approx(expected, abs=0.02)
    bands = [p["band"] for p in body["points"]]
    assert bands == sorted(bands)


def test_max_horizon_limits_the_number_of_requests(fake_backend, monkeypatch):
    monkeypatch.setattr(frontend, "FORECAST_MAX_HORIZON", 21)
    _, body = get_forecast()
    assert [p["horizon_days"] for p in body["points"]] == [7, 7, 14, 21]
    # one request for the weekly model plus one per forward step
    assert len(fake_backend) == 4


def test_user_inputs_are_passed_to_every_step(fake_backend):
    get_forecast("?current_price=171.65&current_demand=88")
    first_round = len(fake_backend)
    assert first_round > 1
    assert all(call["current_price"] == 171.65 for call in fake_backend[:first_round])
    assert all(call["current_demand"] == 88.0 for call in fake_backend[:first_round])
    # a value that is not a number must not break the request body
    get_forecast("?current_price=abc")
    assert all("current_price" not in call for call in fake_backend[first_round:])


def test_single_broken_horizon_does_not_drop_the_curve(fake_backend, monkeypatch):
    def flaky(payload):
        horizon = (pd.Timestamp(payload["date"]) - pd.Timestamp(ORIGIN)).days
        fake_backend.append(dict(payload))
        if horizon == 28:
            return {"error": "no forward feature vector for target_date"}, 404
        return fake_response(payload)

    monkeypatch.setattr(frontend, "_post_predict", flaky)
    status, body = get_forecast()
    assert status == 200
    assert 28 not in [p["horizon_days"] for p in body["points"]]
    assert body["errors"]["forward_28d"]
    assert len(body["points"]) == 12  # weekly + 11 of the 12 forward steps


def test_unreachable_service_reports_one_error_without_retrying(fake_backend, monkeypatch):
    def unreachable(payload):
        fake_backend.append(dict(payload))
        return {"error": "prediction service is unreachable: boom"}, 502

    monkeypatch.setattr(frontend, "_post_predict", unreachable)
    status, body = get_forecast()
    assert status == 200
    assert "error" in body
    # a dead service must cost one timeout, not nine
    assert len(fake_backend) == 1


def test_curve_is_cached_per_origin_and_inputs(fake_backend):
    get_forecast()
    first_count = len(fake_backend)
    get_forecast()
    assert len(fake_backend) == first_count
    get_forecast("?current_price=180")
    assert len(fake_backend) > first_count


# ---------------------------------------------------------------------------
# Measured daily prices inside the forecast window (actual vs. forecast)
# ---------------------------------------------------------------------------

def test_measured_prices_give_one_entry_per_calendar_day():
    """Days without a published price stay in the list as None, so the chart
    keeps one label per day and the measured line only has a hole there."""
    series = {
        "dates": ["2026-10-09", "2026-10-12", "2026-10-13"],
        "price": [100.0, 105.5, None],
    }
    out = frontend._measured_prices(series, "2026-10-09", "2026-10-14")
    assert [entry["date"] for entry in out] == [
        "2026-10-09", "2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14",
    ]
    assert [entry["value"] for entry in out] == [100.0, None, None, 105.5, None, None]


def test_measured_prices_round_to_the_precision_of_the_curve():
    series = {"dates": ["2026-10-09"], "price": [170.9947]}
    assert frontend._measured_prices(series, "2026-10-09", "2026-10-09")[0]["value"] == 170.99


def test_forecast_carries_the_measured_days_of_its_whole_window(fake_backend):
    """The history line must not stop at the origin: every day the curve
    describes is answered with the real price of that day (or None while it is
    still in the future)."""
    _, body = get_forecast()
    measured = body["measured"]
    last_forecast_date = max(p["date"] for p in body["points"])
    assert measured[0]["date"] == ORIGIN
    assert measured[-1]["date"] == last_forecast_date
    # one entry per calendar day, in order, without gaps
    days = [entry["date"] for entry in measured]
    assert days == sorted(days)
    span = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days + 1
    assert span == len(days)
    # the origin day is measured (rounded like the curve), every later day of
    # the curve is not yet
    assert measured[0]["value"] == pytest.approx(body["origin_price"], abs=0.01)
    assert all(entry["value"] is None for entry in measured[1:])


def test_measured_days_are_filled_once_the_history_covers_them(fake_backend):
    """While the curve lies in the future its measured window is empty; the
    moment the price history has published those days, the same window must
    come back filled instead of null."""
    _, body = get_forecast()
    window = [entry["date"] for entry in body["measured"]]
    assert all(entry["value"] is None for entry in body["measured"][1:])
    filled = frontend._measured_prices(
        {"dates": window, "price": [120.0] * len(window)}, window[0], window[-1]
    )
    assert [entry["date"] for entry in filled] == window
    assert [entry["value"] for entry in filled] == [120.0] * len(window)


def test_forecast_without_price_history_is_an_error(monkeypatch):
    monkeypatch.setitem(frontend._price_cache, "data", ({"dates": [], "price": []}, "local"))
    monkeypatch.setitem(frontend._forecast_cache, "data", None)
    status, body = get_forecast()
    assert status == 503
    assert "error" in body