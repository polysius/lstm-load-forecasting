"""Cleaning, the data contract, and versioned snapshots."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from forecast.config import PreprocessingConfig
from forecast.data.cleaning import clean_hourly, fill_short_gaps
from forecast.data.validate import DataContractError, validate
from forecast.data.versioning import (
    DataIntegrityError,
    content_hash,
    latest_snapshot,
    list_snapshots,
    load_snapshot,
    write_snapshot,
)


def hourly(values, start="2024-01-01", tz="UTC") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="h", tz=tz, name="timestamp")
    return pd.DataFrame({"load": np.asarray(values, dtype=float)}, index=idx)


class TestFillShortGaps:
    def test_short_gap_filled_long_gap_left_alone(self):
        s = hourly([1, np.nan, np.nan, 4, 5, np.nan, np.nan, np.nan, np.nan, 10])["load"]
        out = fill_short_gaps(s, max_gap=3)
        np.testing.assert_allclose(out.iloc[:5], [1, 2, 3, 4, 5])
        assert out.iloc[5:9].isna().all(), "a 4-hour outage must not be partially invented"

    def test_edges_are_not_extrapolated(self):
        s = hourly([np.nan, 2, 3, np.nan])["load"]
        out = fill_short_gaps(s, max_gap=3)
        assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[-1])


class TestCleanHourly:
    def test_resamples_trims_and_masks_zeros(self):
        idx = pd.date_range("2024-01-01", periods=16, freq="15min", tz="UTC")
        raw = pd.Series([0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 0, 0, 0, 0], index=idx, dtype=float)
        out = clean_hourly(raw, PreprocessingConfig(), name="load")
        assert list(out.values) == [4.0, 8.0]
        assert out.index.freq is None or out.index.freqstr in ("h", "60min")

    def test_mad_outlier_mask(self):
        values = np.full(100, 10.0) + np.random.default_rng(0).normal(0, 0.1, 100)
        values[50] = 1e6
        raw = pd.Series(values, index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"))
        out = clean_hourly(raw, PreprocessingConfig(outlier_mad_threshold=8.0), name="load")
        assert out.iloc[50] < 100  # masked, then the 1-hour hole is interpolated


class TestContract:
    def test_valid_frame_passes(self):
        summary = validate(hourly(np.arange(48)))
        assert summary["n_rows"] == 48 and summary["n_gaps"] == 0

    @pytest.mark.parametrize(
        "frame",
        [
            hourly([1, 2, 3]).tz_localize(None),
            hourly([1, -2, 3]),
            hourly([1, 2, 3]).iloc[[0, 2]],
            hourly([1, 2, 3]).iloc[::-1],
        ],
        ids=["naive-tz", "negative", "dropped-row", "unsorted"],
    )
    def test_violations_raise(self, frame):
        with pytest.raises(DataContractError):
            validate(frame)

    def test_gap_accounting(self):
        summary = validate(hourly([1, np.nan, np.nan, 4, np.nan, 6]))
        assert (summary["n_gaps"], summary["total_gap_hours"], summary["max_gap_hours"]) == (
            2,
            3,
            2,
        )


class TestVersioning:
    def test_hash_is_stable_across_parquet_roundtrip(self, tmp_path):
        df = hourly(np.random.default_rng(0).normal(size=100), tz="Europe/Lisbon")
        df.to_parquet(tmp_path / "x.parquet")
        assert content_hash(pd.read_parquet(tmp_path / "x.parquet")) == content_hash(df)

    def test_hash_detects_any_change(self):
        df = hourly(np.arange(10))
        changed = df.copy()
        changed.iloc[3, 0] = 3.0000001
        assert content_hash(df) != content_hash(changed)

    def test_dated_snapshots_and_dedup(self, tmp_path):
        t1, t2 = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)
        a = write_snapshot(hourly(np.arange(10)), dataset="d", data_dir=tmp_path, now=t1)
        same = write_snapshot(hourly(np.arange(10)), dataset="d", data_dir=tmp_path, now=t2)
        assert same.version == a.version == "2026-01-01T000000Z"
        b = write_snapshot(hourly(np.arange(11)), dataset="d", data_dir=tmp_path, now=t2)
        assert [s.version for s in list_snapshots("d", tmp_path)] == [a.version, b.version]
        assert latest_snapshot("d", tmp_path).sha256 == b.sha256

    def test_tampering_is_detected(self, tmp_path):
        snap = write_snapshot(hourly(np.arange(10)), dataset="d", data_dir=tmp_path)
        hourly(np.arange(10) * 2).to_parquet(snap.path)
        with pytest.raises(DataIntegrityError):
            load_snapshot(snap)
