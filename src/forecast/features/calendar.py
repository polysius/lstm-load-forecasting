"""Calendar features derived from timestamps only, so they carry no leakage risk."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour, weekday, month, weekend flag and their cyclical sin/cos encodings."""
    hour, dow, month = index.hour, index.day_of_week, index.month
    return pd.DataFrame(
        {
            "hour": hour,
            "dayofweek": dow,
            "month": month,
            "is_weekend": (dow >= 5).astype(np.int8),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        },
        index=index,
    )


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append calendar columns to a frame with a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("requires a DatetimeIndex")
    return pd.concat([df, calendar_frame(df.index)], axis=1)


CYCLICAL = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")


def cyclical_covariates(index: pd.DatetimeIndex) -> np.ndarray:
    """``(n, 4)`` hour/day-of-week sin-cos encodings, the covariates the LSTM can use."""
    return calendar_frame(index)[list(CYCLICAL)].to_numpy(dtype=np.float64)
