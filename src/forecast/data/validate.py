"""Data contract for every processed series, whatever its source."""

from __future__ import annotations

from typing import Any

import pandas as pd


class DataContractError(ValueError):
    """The processed data violates the contract the models rely on."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise DataContractError(msg)


def validate(df: pd.DataFrame, freq: str = "1h") -> dict[str, Any]:
    """Run all contract checks; return a summary, raise ``DataContractError`` on violation."""
    _require(isinstance(df.index, pd.DatetimeIndex), "index must be a DatetimeIndex")
    _require(df.index.tz is not None, "index must be timezone-aware")
    _require(df.index.is_monotonic_increasing, "index must be monotonically increasing")
    _require(df.index.is_unique, "index must have unique timestamps")
    _require(len(df) > 0, "frame is empty")

    expected = pd.date_range(df.index[0], df.index[-1], freq=freq, tz=df.index.tz)
    missing = expected.difference(df.index)
    _require(
        len(missing) == 0,
        f"non-uniform sampling: {len(missing)} timestamps missing; gaps must be explicit NaN",
    )

    for col in df.columns:
        _require(pd.api.types.is_numeric_dtype(df[col]), f"column {col!r} must be numeric")
        n_negative = int((df[col] < 0).sum())
        _require(n_negative == 0, f"column {col!r} has {n_negative} negative values")

    n_nan = int(df.isna().sum().sum())
    summary: dict[str, Any] = {
        "tz": str(df.index.tz),
        "freq": freq,
        "n_rows": len(df),
        "n_nan": n_nan,
        "nan_pct": round(n_nan / (len(df) * max(len(df.columns), 1)) * 100, 4),
        "start": df.index[0].isoformat(),
        "end": df.index[-1].isoformat(),
    }
    if len(df.columns) == 1:
        is_nan = df.iloc[:, 0].isna()
        run_id = (is_nan != is_nan.shift(fill_value=False)).cumsum()
        gap_lengths = is_nan.groupby(run_id).sum()
        gap_lengths = gap_lengths[gap_lengths > 0]
        summary["n_gaps"] = len(gap_lengths)
        summary["total_gap_hours"] = int(gap_lengths.sum())
        summary["max_gap_hours"] = int(gap_lengths.max()) if len(gap_lengths) else 0
    return summary
