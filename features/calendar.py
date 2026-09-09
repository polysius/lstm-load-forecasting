"""Calendar features derived from timestamps only — no leakage risk."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day, day-of-week, month, and holiday flag to a DatetimeIndex DataFrame."""
    assert isinstance(df.index, pd.DatetimeIndex), "Requires DatetimeIndex"

    out = df.copy()
    out["hour"] = df.index.hour
    out["dayofweek"] = df.index.dayofweek  # 0=Monday
    out["month"] = df.index.month
    out["is_weekend"] = (df.index.dayofweek >= 5).astype(np.int8)

    # Cyclical encoding for hour and dayofweek (helps neural nets)
    out["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (df.index.month - 1) / 12)

    return out
