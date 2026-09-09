"""Preprocess raw UCI electricity data into a clean hourly parquet file."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "config" / "data.yaml"


def load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw UCI electricity txt file.

    The file uses semicolons as separators and commas as decimal points.
    """
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
    )
    df.index.name = "timestamp"
    return df


def preprocess(raw_path: Path | None = None) -> pd.DataFrame:
    cfg = load_config()

    if raw_path is None:
        raw_path = _ROOT / "data" / "raw" / cfg["dataset"]["raw_filename"]

    if not raw_path.exists():
        print(f"Raw file not found: {raw_path}")
        print("Run data/download.py first.")
        sys.exit(1)

    print(f"Loading raw data from {raw_path} ...")
    df_raw = load_raw(raw_path)
    print(f"  Shape: {df_raw.shape}, range: {df_raw.index[0]} to {df_raw.index[-1]}")

    target_client = cfg["dataset"]["target_client"]
    if target_client not in df_raw.columns:
        available = [c for c in df_raw.columns[:10]]
        raise ValueError(
            f"Client '{target_client}' not in dataset. First 10 columns: {available}"
        )

    series = df_raw[target_client].copy()
    series.name = "load_kw"

    # Localize timestamps — the raw data uses local Portuguese time but has DST
    # ambiguities. Localize as UTC first (preserving numeric values), then convert.
    tz = cfg["preprocessing"]["timezone"]
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC").tz_convert(tz)
    else:
        series.index = series.index.tz_convert(tz)

    # Replace zeros with NaN (meter off or missing data — common in this dataset)
    n_zeros = (series == 0).sum()
    if n_zeros > 0:
        print(f"  Replacing {n_zeros} zero readings with NaN (meter off)")
        series = series.replace(0, np.nan)

    # Resample to hourly
    freq = cfg["preprocessing"]["resample_freq"]
    print(f"  Resampling to {freq} ...")
    series_hourly = series.resample(freq).mean()

    # Detect outliers (report but don't remove)
    q1 = series_hourly.quantile(0.01)
    q99 = series_hourly.quantile(0.99)
    iqr = q99 - q1
    lower = q1 - 3 * iqr
    upper = q99 + 3 * iqr
    n_outliers = ((series_hourly < lower) | (series_hourly > upper)).sum()
    print(f"  Outliers (3*IQR beyond 1st/99th percentile): {n_outliers}")

    # Build a complete hourly index — gaps become explicit NaN
    full_idx = pd.date_range(
        series_hourly.index[0], series_hourly.index[-1], freq=freq, tz=series_hourly.index.tz
    )
    series_hourly = series_hourly.reindex(full_idx)
    series_hourly.index.name = "timestamp"

    n_nan_before = series_hourly.isna().sum()
    print(f"  NaN values before interpolation: {n_nan_before}")

    # Interpolate small gaps only (within max_gap limit)
    max_gap = cfg["preprocessing"]["interpolate_max_gap_hours"]
    series_hourly = series_hourly.interpolate(
        method=cfg["preprocessing"]["interpolate_method"],
        limit=max_gap,
    )

    n_nan_after = series_hourly.isna().sum()
    print(f"  NaN values after interpolation (gaps > {max_gap}h remain): {n_nan_after}")

    # Drop leading/trailing NaN rows (e.g. MT_001 has no data for all of 2011)
    first_valid = series_hourly.first_valid_index()
    last_valid = series_hourly.index[series_hourly.notna()][-1]
    if first_valid is not None:
        n_before = len(series_hourly)
        series_hourly = series_hourly.loc[first_valid:last_valid]
        n_dropped = n_before - len(series_hourly)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} leading/trailing NaN rows")
            print(f"  Usable range: {first_valid} to {last_valid}")

    df_out = series_hourly.to_frame()

    # Save
    out_dir = _ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "electricity_hourly.parquet"
    df_out.to_parquet(out_path)
    print(f"  Saved: {out_path} ({len(df_out)} rows)")

    return df_out


if __name__ == "__main__":
    preprocess()
