"""Shared cleaning for any raw load series: resample, make gaps explicit, fill short gaps."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from forecast.config import PreprocessingConfig

log = logging.getLogger(__name__)


def fill_short_gaps(series: pd.Series, max_gap: int, method: str = "linear") -> pd.Series:
    """Interpolate runs of NaN no longer than ``max_gap``; longer gaps stay NaN.

    ``Series.interpolate(limit=n)`` alone would also fill the first ``n`` points of
    a *long* gap, which silently invents data inside outages.
    """
    if max_gap <= 0 or not series.isna().any():
        return series
    is_nan = series.isna()
    run_id = (is_nan != is_nan.shift(fill_value=False)).cumsum()
    run_len = is_nan.groupby(run_id).transform("sum")
    short = is_nan & (run_len <= max_gap)
    filled = series.interpolate(method=method, limit_area="inside")
    return series.where(~short, filled)


def mad_outliers(series: pd.Series, threshold: float) -> pd.Series:
    """Boolean mask of points more than ``threshold`` robust z-scores from the median."""
    med = series.median()
    mad = (series - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=series.index)
    robust_z = 0.6745 * (series - med).abs() / mad
    return robust_z > threshold


def clean_hourly(series: pd.Series, pp: PreprocessingConfig, *, name: str) -> pd.Series:
    """Raw readings -> uniform hourly series with explicit NaN gaps and trimmed edges."""
    series = series.copy()
    series.name = name

    if pp.zero_is_missing:
        n_zeros = int((series == 0).sum())
        if n_zeros:
            log.info("replacing %d zero readings with NaN (meter off)", n_zeros)
            series = series.replace(0, np.nan)

    hourly = series.resample(pp.resample_freq).mean()

    if pp.outlier_mad_threshold is not None:
        mask = mad_outliers(hourly, pp.outlier_mad_threshold)
        if mask.any():
            log.info(
                "masking %d outliers (> %.1f robust z)", int(mask.sum()), pp.outlier_mad_threshold
            )
            hourly = hourly.mask(mask)

    full_idx = pd.date_range(
        hourly.index[0], hourly.index[-1], freq=pp.resample_freq, tz=hourly.index.tz
    )
    hourly = hourly.reindex(full_idx)
    hourly.index.name = "timestamp"
    log.info("NaN before gap filling: %d", int(hourly.isna().sum()))

    hourly = fill_short_gaps(hourly, pp.interpolate_max_gap_hours, pp.interpolate_method)
    log.info(
        "NaN after filling gaps <= %dh: %d", pp.interpolate_max_gap_hours, int(hourly.isna().sum())
    )

    valid = hourly.notna().to_numpy()
    if not valid.any():
        raise ValueError("series is entirely missing after cleaning")
    first, last = hourly.index[int(np.argmax(valid))], hourly.index[valid][-1]
    trimmed = hourly.loc[first:last]
    if len(trimmed) < len(hourly):
        log.info(
            "trimmed %d leading/trailing NaN rows; usable %s -> %s",
            len(hourly) - len(trimmed),
            first,
            last,
        )
    return trimmed
