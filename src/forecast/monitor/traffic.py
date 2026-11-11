"""Simulated production traffic: the latest week of live EIA demand -> POST /forecast."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from forecast.config import DataConfig
from forecast.data.cleaning import fill_short_gaps

log = logging.getLogger(__name__)


def latest_window(series: pd.Series, hours: int = 168) -> tuple[list[float], datetime]:
    """The most recent ``hours`` of contiguous, gap-free hourly values."""
    s = series.sort_index()
    full = pd.date_range(s.index[0], s.index[-1], freq="1h", tz=s.index.tz)
    s = fill_short_gaps(s.reindex(full), max_gap=3)
    s = s.loc[: s.last_valid_index()]
    window = s.iloc[-hours:]
    if len(window) < hours or window.isna().any():
        raise ValueError(f"no gap-free {hours} h window ending at {s.index[-1]}")
    return window.to_numpy(dtype=float).round(3).tolist(), window.index[-1].to_pydatetime()


def post_forecast(
    url: str,
    history: list[float],
    last_timestamp: datetime,
    *,
    series_id: str,
    horizon: int = 24,
    timeout: float = 30.0,
    session=None,
) -> dict[str, Any]:
    import requests

    session = session or requests.Session()
    resp = session.post(
        url.rstrip("/") + "/forecast",
        json={
            "series_id": series_id,
            "history": history,
            "last_timestamp": last_timestamp.isoformat(),
            "horizon": horizon,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def send_live_traffic(cfg: DataConfig, url: str) -> dict[str, Any]:
    from forecast.data.eia import fetch_recent

    raw = fetch_recent(cfg, hours=cfg.task.input_window + 96)  # slack for publication lag
    history, last = latest_window(raw, cfg.task.input_window)
    body = post_forecast(url, history, last, series_id=cfg.eia.respondent if cfg.eia else "")
    log.info(
        "forecast from %s: model v%s, next hour %.0f %s [%.0f, %.0f]",
        last,
        body["model_version"],
        body["forecasts"][0]["point"],
        body["units"],
        body["forecasts"][0]["lower"],
        body["forecasts"][0]["upper"],
    )
    return {
        "origin": last.isoformat(),
        "model_version": body["model_version"],
        "request_id": body["request_id"],
        "first_point": float(np.round(body["forecasts"][0]["point"], 1)),
    }
