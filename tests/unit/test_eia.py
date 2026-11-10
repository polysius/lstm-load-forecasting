"""EIA client without the network: pagination, parsing, cleaning, key redaction."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import pytest

from forecast.config import load_data_config
from forecast.data import eia

KEY = "sk-test-0123456789abcdef"


class FakeResponse:
    def __init__(self, status: int, body: dict | None = None, text: str = ""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


class FakeSession:
    """Serves ``rows`` in pages, honouring offset/length like the real API."""

    def __init__(self, rows, status=200, text=""):
        self.rows, self.status, self.text, self.calls = rows, status, text, []

    def get(self, url, params, timeout):
        self.calls.append(params)
        if self.status != 200:
            return FakeResponse(self.status, text=self.text)
        off, n = params["offset"], params["length"]
        return FakeResponse(
            200, {"response": {"total": str(len(self.rows)), "data": self.rows[off : off + n]}}
        )


def rows(n, start="2026-01-01T00", value=90000.0):
    t0 = pd.Timestamp(start, tz="UTC")
    return [
        {
            "period": (t0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H"),
            "respondent": "PJM",
            "type": "D",
            "value": str(value + i),
        }
        for i in range(n)
    ]


@pytest.fixture
def cfg():
    base = load_data_config("eia")
    return base.model_copy(update={"eia": base.eia.model_copy(update={"page_size": 5})})


def test_paginates_and_parses(cfg):
    session = FakeSession(rows(12))
    s = eia.fetch_range(
        cfg,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        session=session,
        key=KEY,
    )
    assert [c["offset"] for c in session.calls] == [0, 5, 10]
    assert len(s) == 12 and s.iloc[0] == 90000.0 and str(s.index.tz) == "UTC"
    assert s.index[1] - s.index[0] == pd.Timedelta(hours=1)
    assert session.calls[0]["facets[respondent][]"] == "PJM"


def test_errors_never_leak_the_key(cfg):
    session = FakeSession([], status=403, text=f"invalid api_key={KEY} for this route")
    with pytest.raises(eia.EIAError) as err:
        eia.fetch_range(
            cfg,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            session=session,
            key=KEY,
        )
    assert KEY not in str(err.value) and "api_key=***" in str(err.value)


def test_urllib3_retry_warnings_never_leak_the_key(caplog):
    pytest.importorskip("requests")
    eia._session()
    caplog.set_level(logging.WARNING, logger="urllib3")
    # The message urllib3.connectionpool logs before retrying a broken connection.
    logging.getLogger("urllib3.connectionpool").warning(
        "Retrying (%r) after connection broken by '%r': %s",
        "Retry(total=4, connect=None, read=None, redirect=None, status=None)",
        "ReadTimeoutError(\"HTTPSConnectionPool(host='api.eia.gov', port=443)\")",
        f"/v2/electricity/rto/region-data/data/?frequency=hourly&api_key={KEY}&offset=0",
    )
    assert "Retrying" in caplog.text
    assert KEY not in caplog.text and "api_key=***" in caplog.text


def test_missing_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    monkeypatch.setenv("FORECAST_ROOT", str(tmp_path))  # no .env there
    with pytest.raises(eia.MissingAPIKey, match="EIA_API_KEY"):
        eia.api_key()


def test_public_source_has_no_key(cfg):
    assert "api_key" not in eia.public_source(cfg)


def test_history_is_cleaned_to_the_contract(cfg, monkeypatch):
    data = rows(60)
    data[10]["value"] = None  # a 1-hour hole: interpolated
    data[20]["value"] = "0"  # a zero reading: treated as missing, then interpolated
    monkeypatch.setattr(eia, "api_key", lambda: KEY)
    monkeypatch.setattr(eia, "_session", lambda: FakeSession(data))
    from forecast.data.validate import validate

    df, extra = eia.fetch_history(cfg, end=datetime(2026, 1, 3, 12, tzinfo=UTC))
    assert validate(df)["n_nan"] == 0
    assert extra["respondent"] == "PJM" and extra["raw_rows"] == 59
