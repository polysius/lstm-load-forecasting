"""NumPy-only calendar encodings, shared by training and the (pandas-free) serving path.

Using one implementation on both sides is what rules out train/serve skew.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

AVAILABLE = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")
_HOUR_NS = 3_600_000_000_000


def _to_utc_naive(t: datetime) -> datetime:
    if getattr(t, "tzinfo", None) is not None:
        return t.astimezone(UTC).replace(tzinfo=None)
    return t


def epoch_hours(timestamps) -> np.ndarray:
    """Hours since 1970-01-01 UTC for datetime64 values, datetimes or a DatetimeIndex."""
    if hasattr(timestamps, "tz_convert"):  # pandas DatetimeIndex
        if timestamps.tz is not None:
            timestamps = timestamps.tz_convert("UTC").tz_localize(None)
        arr = np.asarray(timestamps, dtype="datetime64[ns]")
    else:
        arr = np.asarray(timestamps)
        if arr.dtype == object:
            arr = np.array([np.datetime64(_to_utc_naive(t), "ns") for t in arr.ravel()])
    return arr.astype("datetime64[ns]").astype(np.int64) // _HOUR_NS


def hourly_range(last: datetime, n: int) -> np.ndarray:
    """``n`` consecutive epoch-hours ending at ``last`` (inclusive)."""
    end = int(epoch_hours(np.array([last], dtype=object))[0])
    return np.arange(end - n + 1, end + 1, dtype=np.int64)


def encode(hours_since_epoch: np.ndarray, names: tuple[str, ...] | list[str]) -> np.ndarray:
    """``(n, len(names))`` sin/cos encodings of hour-of-day and day-of-week (UTC)."""
    h = np.asarray(hours_since_epoch, dtype=np.int64)
    unknown = set(names) - set(AVAILABLE)
    if unknown:
        raise ValueError(f"unknown covariates {sorted(unknown)}; available: {AVAILABLE}")
    if not names:
        return np.empty((len(h), 0))
    hour = h % 24
    dow = ((h // 24) + 3) % 7  # 1970-01-01 was a Thursday; Monday == 0
    cols = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }
    return np.column_stack([cols[n] for n in names])
