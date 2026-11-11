"""EIA API v2 client for EIA-930 hourly balancing-authority demand.

The API key travels as a query parameter, so every error message is redacted before
it can leave this module, and the session redacts urllib3's log records, whose
retry warnings and request lines carry the full URL.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from forecast.config import DataConfig, project_root
from forecast.data.cleaning import clean_hourly

log = logging.getLogger(__name__)

_KEY_RE = re.compile(r"(api_key=)[^&\s]+", re.IGNORECASE)


class MissingAPIKey(RuntimeError):
    pass


class EIAError(RuntimeError):
    pass


def redact(text: str) -> str:
    return _KEY_RE.sub(r"\1***", text)


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


_REDACT_FILTER = _RedactFilter()
# Filters on a logger do not apply to records from its children, so name each one.
_URLLIB3_LOGGERS = ("urllib3.connectionpool", "urllib3.util.retry")


def api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(project_root() / ".env", override=False)
    except ImportError:  # pragma: no cover - python-dotenv is in the data extra
        pass
    key = os.environ.get("EIA_API_KEY", "").strip()
    if not key:
        raise MissingAPIKey(
            "EIA_API_KEY is not set: add it to .env (see .env.example) or the environment"
        )
    return key


def public_source(cfg: DataConfig) -> str:
    """The query without the key, recorded in manifests and MLflow."""
    assert cfg.eia is not None
    e = cfg.eia
    return (
        f"{e.base_url}?frequency=hourly&data[0]=value"
        f"&facets[respondent][]={e.respondent}&facets[type][]={e.series_type}"
    )


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    for name in _URLLIB3_LOGGERS:
        logger = logging.getLogger(name)
        if _REDACT_FILTER not in logger.filters:
            logger.addFilter(_REDACT_FILTER)
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "lstm-load-forecasting/0.2 (research; python-requests)"
    return s


def fetch_range(
    cfg: DataConfig, start: datetime, end: datetime, *, session=None, key: str | None = None
) -> pd.Series:
    """Raw hourly values in ``[start, end]`` (UTC, period labels), paginated."""
    import requests

    assert cfg.eia is not None
    e = cfg.eia
    key = key or api_key()
    session = session or _session()
    params: dict[str, Any] = {
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": e.respondent,
        "facets[type][]": e.series_type,
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": e.page_size,
    }
    rows: list[dict[str, Any]] = []
    offset, total = 0, None
    while total is None or offset < total:
        try:
            resp = session.get(
                e.base_url, params={**params, "offset": offset, "api_key": key}, timeout=(10, 60)
            )
        except requests.RequestException as exc:
            raise EIAError(redact(f"EIA request failed: {exc}")) from None
        if resp.status_code != 200:
            raise EIAError(redact(f"EIA API returned {resp.status_code}: {resp.text[:300]}"))
        body = resp.json().get("response", {})
        page = body.get("data", [])
        total = int(body.get("total", 0) or 0)
        rows.extend(page)
        if not page:
            break
        offset += len(page)
    log.info(
        "EIA %s/%s: %d rows %s -> %s",
        e.respondent,
        e.series_type,
        len(rows),
        params["start"],
        params["end"],
    )
    if not rows:
        return pd.Series(dtype="float64", name=cfg.target_col)
    frame = pd.DataFrame(rows)
    ts = pd.to_datetime(frame["period"], format="%Y-%m-%dT%H", utc=True)
    values = pd.to_numeric(frame["value"], errors="coerce")
    series = pd.Series(values.to_numpy(dtype="float64"), index=ts, name=cfg.target_col)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.index.name = "timestamp"
    return series


def fetch_history(
    cfg: DataConfig, *, end: datetime | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``history_days`` of cleaned hourly demand ending at the latest complete hour."""
    assert cfg.eia is not None
    end = (end or datetime.now(UTC)).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=cfg.eia.history_days)
    raw = fetch_range(cfg, start, end)
    if raw.empty:
        raise EIAError(f"EIA returned no data for {cfg.eia.respondent} {start} -> {end}")
    clean = clean_hourly(raw, cfg.preprocessing, name=cfg.target_col)
    extra = {
        "respondent": cfg.eia.respondent,
        "series_type": cfg.eia.series_type,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "raw_rows": int(raw.notna().sum()),
    }
    return clean.to_frame(), extra


def fetch_recent(cfg: DataConfig, hours: int, *, end: datetime | None = None) -> pd.Series:
    """The last ``hours`` of raw (uncleaned) hourly values, for live traffic and actuals."""
    end = (end or datetime.now(UTC)).replace(minute=0, second=0, microsecond=0)
    return fetch_range(cfg, end - timedelta(hours=hours), end)
