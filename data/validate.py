"""Validate the preprocessed dataset against the data contract (SPEC §3.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "config" / "data.yaml"


def load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


def validate(df: pd.DataFrame, freq: str = "1h") -> dict[str, object]:
    """Run all contract checks. Returns a summary dict. Raises on violation."""
    results: dict[str, object] = {}

    # 1. Timestamp index is monotonic, timezone-aware, and unique
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert df.index.tz is not None, "Index must be timezone-aware"
    assert df.index.is_monotonic_increasing, "Index must be monotonically increasing"
    assert df.index.is_unique, "Index must have unique timestamps"
    results["tz"] = str(df.index.tz)
    results["monotonic"] = True
    results["unique"] = True

    # 2. Sampling frequency is uniform after resampling; gaps are explicit NaN
    expected_idx = pd.date_range(df.index[0], df.index[-1], freq=freq, tz=df.index.tz)
    missing_timestamps = expected_idx.difference(df.index)
    if len(missing_timestamps) > 0:
        raise AssertionError(
            f"Non-uniform sampling: {len(missing_timestamps)} expected timestamps missing. "
            f"Gaps must be explicit NaN rows, not dropped."
        )
    results["freq"] = freq
    results["n_rows"] = len(df)

    # 3. Target column is numeric and non-negative
    target_col = [c for c in df.columns if c != "timestamp"]
    for col in target_col:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} must be numeric"
        n_negative = (df[col] < 0).sum()
        if n_negative > 0:
            raise AssertionError(f"Column {col} has {n_negative} negative values")

    # 4. Missing value accounting
    n_nan = int(df.isna().sum().sum())
    nan_pct = n_nan / (len(df) * len(target_col)) * 100
    results["n_nan"] = n_nan
    results["nan_pct"] = round(nan_pct, 4)

    # Gap analysis: consecutive NaN runs
    if len(target_col) == 1:
        col = target_col[0]
        is_nan = df[col].isna()
        gap_starts = is_nan & ~is_nan.shift(1, fill_value=False)
        gap_ends = is_nan & ~is_nan.shift(-1, fill_value=False)
        n_gaps = int(gap_starts.sum())
        if n_gaps > 0:
            gap_lengths = []
            starts = df.index[gap_starts]
            ends = df.index[gap_ends]
            for s, e in zip(starts, ends, strict=True):
                gap_lengths.append((e - s).total_seconds() / 3600 + 1)
            total_gap_hours = sum(gap_lengths)
            results["n_gaps"] = n_gaps
            results["total_gap_hours"] = round(total_gap_hours, 1)
            results["max_gap_hours"] = round(max(gap_lengths), 1)
        else:
            results["n_gaps"] = 0
            results["total_gap_hours"] = 0
            results["max_gap_hours"] = 0

    results["date_range"] = f"{df.index[0]} to {df.index[-1]}"
    results["passed"] = True
    return results


def validate_file(path: Path) -> dict[str, object]:
    """Load a parquet file and validate it."""
    df = pd.read_parquet(path)
    cfg = load_config()
    freq = cfg["preprocessing"]["resample_freq"]
    print(f"Validating {path} ...")
    results = validate(df, freq=freq)
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def write_snapshot(results: dict[str, object], dest: Path | None = None) -> None:
    """Write SNAPSHOT.md from validation results."""
    if dest is None:
        dest = _ROOT / "data" / "SNAPSHOT.md"

    cfg = load_config()
    lines = [
        "# Data Snapshot",
        "",
        f"**Source:** {cfg['dataset']['source_url']}",
        "**Download date:** recorded at download time",
        f"**Target client:** {cfg['dataset']['target_client']}",
        "",
        "## Validated properties",
        "",
        f"- **Rows:** {results.get('n_rows', 'N/A')}",
        f"- **Date range:** {results.get('date_range', 'N/A')}",
        f"- **Frequency:** {results.get('freq', 'N/A')}",
        f"- **Timezone:** {results.get('tz', 'N/A')}",
        "",
        "## Missing data",
        "",
        f"- **NaN values:** {results.get('n_nan', 'N/A')} ({results.get('nan_pct', 'N/A')}%)",
        f"- **Gaps:** {results.get('n_gaps', 'N/A')}",
        f"- **Total gap hours:** {results.get('total_gap_hours', 'N/A')}",
        f"- **Max gap hours:** {results.get('max_gap_hours', 'N/A')}",
        "",
        "## Preprocessing",
        "",
        f"- Resampled to {cfg['preprocessing']['resample_freq']}",
        f"- Interpolation: {cfg['preprocessing']['interpolate_method']}, "
        f"max gap {cfg['preprocessing']['interpolate_max_gap_hours']}h",
        f"- Gaps > {cfg['preprocessing']['interpolate_max_gap_hours']}h left as NaN",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    processed = _ROOT / "data" / "processed" / "electricity_hourly.parquet"
    if not processed.exists():
        print(f"File not found: {processed}")
        print("Run data/preprocess.py first.")
        sys.exit(1)
    results = validate_file(processed)
    write_snapshot(results)
