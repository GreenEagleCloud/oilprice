"""Price collector resilience: the heizoel24 fetch must survive a bad upstream.

The 2026-10-25..27 outage was this call failing fast and repeatedly (Job
`collect-daily-prices-29885730`: 7 attempts in 5m43s, seven S3 reads, no write),
while `fetch_prices` had no retry and no check of what the API actually returned.
These tests pin the replacement: bounded retries with backoff, a payload that is
validated instead of trusted, and a loud log line when the feed stops serving
new days - the situation that left the frontend showing week-old prices.
"""

import io
import pathlib
import sys

import pandas as pd
import pytest
import requests
from botocore.exceptions import ClientError

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import collect_daily_prices as cdp  # noqa: E402


def payload(*dates, demand=True):
    rows = []
    for day in dates:
        row = {"DateTime": f"{day}T00:00:00", "Price": 94.9}
        if demand:
            row["Demand"] = 1.5
        rows.append(row)
    return rows


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        return self._body


class FakeGet:
    """Answers with the queued responses/exceptions; the last one repeats."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def waits():
    """Retries are tested without sleeping, and every backoff is recorded."""
    recorded = []
    return recorded, recorded.append


# ------------------------------------------------------------- retries ----
def test_transient_http_error_is_retried(monkeypatch, waits):
    recorded, record = waits
    fake = FakeGet(FakeResponse(503), FakeResponse(200, payload("2026-10-27")))
    monkeypatch.setattr(cdp.requests, "get", fake)

    frame = cdp.fetch_prices(1, attempts=3, sleep=record)

    assert len(fake.calls) == 2
    assert recorded == [1]
    assert list(frame["date"]) == ["2026-10-27"]


def test_connection_error_is_retried(monkeypatch, waits):
    recorded, record = waits
    fake = FakeGet(requests.ConnectionError("reset"), FakeResponse(200, payload("2026-10-27")))
    monkeypatch.setattr(cdp.requests, "get", fake)

    frame = cdp.fetch_prices(1, attempts=3, sleep=record)

    assert len(frame) == 1
    assert recorded == [1]


def test_gives_up_after_the_attempt_budget(monkeypatch, waits):
    recorded, record = waits
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(500)))

    with pytest.raises(RuntimeError, match="unavailable after 4 attempts"):
        cdp.fetch_prices(1, attempts=4, sleep=record)

    # exponential backoff, and no sleep after the final failed attempt
    assert recorded == [1, 2, 4]


def test_backoff_is_capped(monkeypatch, waits):
    recorded, record = waits
    monkeypatch.setattr(cdp, "FETCH_MAX_BACKOFF", 8)
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(500)))

    with pytest.raises(RuntimeError, match="unavailable after 6 attempts"):
        cdp.fetch_prices(1, attempts=6, sleep=record)

    assert recorded == [1, 2, 4, 8, 8]


# ------------------------------------------------------------ payload ----
def test_empty_payload_is_retried_and_then_raises(monkeypatch, waits):
    recorded, record = waits
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(200, [])))

    with pytest.raises(RuntimeError, match="unexpected payload: list with 0 rows"):
        cdp.fetch_prices(1, attempts=2, sleep=record)


def test_error_object_instead_of_rows_is_rejected(monkeypatch, waits):
    recorded, record = waits
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(200, {"error": "rate limited"})))

    with pytest.raises(RuntimeError, match="unexpected payload: dict"):
        cdp.fetch_prices(1, attempts=1, sleep=record)


def test_rows_without_the_required_keys_are_rejected(monkeypatch, waits):
    recorded, record = waits
    body = [{"DateTime": "2026-10-27T00:00:00", "Amount": 94.9}]
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(200, body)))

    with pytest.raises(RuntimeError, match=r"lack \['Price'\]"):
        cdp.fetch_prices(1, attempts=1, sleep=record)


def test_unusable_rows_are_dropped_not_crashed(monkeypatch, waits):
    recorded, record = waits
    body = [
        {"DateTime": "2026-10-26T00:00:00", "Price": 94.9, "Demand": 1.5},
        {"DateTime": "2026-10-27T00:00:00", "Price": None, "Demand": 1.5},
        {"DateTime": "not-a-date", "Price": 95.0, "Demand": 1.5},
    ]
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(200, body)))

    frame = cdp.fetch_prices(1, attempts=1, sleep=record)

    assert list(frame["date"]) == ["2026-10-26"]


def test_demand_column_is_optional(monkeypatch, waits):
    recorded, record = waits
    body = payload("2026-10-27", demand=False)
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(200, body)))

    frame = cdp.fetch_prices(1, attempts=1, sleep=record)

    assert list(frame.columns) == ["date", "price", "demand"]
    assert pd.isna(frame["demand"].iloc[0])


# ------------------------------------------------------------ staleness ----
def test_stale_feed_is_reported(capsys):
    frame = pd.DataFrame({"date": ["2026-10-20", "2026-10-24"]})
    age = cdp.warn_if_stale(frame, today=cdp.date(2026, 10, 28), max_age_days=3)

    assert age == 4
    assert "looks stale" in capsys.readouterr().out


def test_fresh_feed_is_not_reported(capsys):
    frame = pd.DataFrame({"date": ["2026-10-27", "2026-10-28"]})
    cdp.warn_if_stale(frame, today=cdp.date(2026, 10, 28), max_age_days=3)

    assert "looks stale" not in capsys.readouterr().out


# ----------------------------------------------------------------- main ----
class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.saved = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.saved[Key] = Body


@pytest.fixture
def fake_s3(monkeypatch):
    existing = pd.DataFrame(
        {"date": ["2026-10-22", "2026-10-23", "2026-10-24"],
         "price": [93.0, 93.5, 94.0],
         "demand": [1.0, 1.1, 1.2]}
    )
    buffer = io.BytesIO()
    existing.to_csv(buffer, index=False)
    client = FakeS3({cdp.KEY: buffer.getvalue()})
    monkeypatch.setattr(cdp.boto3, "client", lambda service: client)
    return client


@pytest.fixture
def fixed_today(monkeypatch):
    class FixedDate(cdp.date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 28)

    monkeypatch.setattr(cdp, "date", FixedDate)


def test_main_merges_fresh_rows_and_reports_the_newest(monkeypatch, fake_s3, fixed_today, capsys, waits):
    recorded, record = waits
    fake = FakeGet(FakeResponse(500), FakeResponse(200, payload("2026-10-26", "2026-10-27", "2026-10-28")))
    monkeypatch.setattr(cdp.requests, "get", fake)
    monkeypatch.setattr(cdp.time, "sleep", record)

    cdp.main()

    saved = pd.read_csv(io.BytesIO(fake_s3.saved[cdp.KEY]))
    out = capsys.readouterr().out
    assert len(saved) == 6
    assert list(saved["date"])[-3:] == ["2026-10-26", "2026-10-27", "2026-10-28"]
    # the retry happens for the real run too, not only when fetch_prices is called directly
    assert len(fake.calls) == 2
    assert recorded == [1]
    assert "newest 2026-10-28" in out
    assert "looks stale" not in out


def test_main_warns_when_the_feed_adds_nothing(monkeypatch, fake_s3, fixed_today, capsys):
    monkeypatch.setattr(cdp.requests, "get", FakeGet(FakeResponse(200, payload("2026-10-23", "2026-10-24"))))

    cdp.main()

    out = capsys.readouterr().out
    assert "no new price records" in out
    assert cdp.KEY in fake_s3.saved
