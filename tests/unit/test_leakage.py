"""Leakage prevention tests (SPEC §5): chronological folds, purge gap, windows, scaling."""

from __future__ import annotations

import numpy as np
import pytest

from forecast.evaluate.splits import generate_folds


class TestChronologicalSplitting:
    """§5.1 — No shuffling, chronological order only."""

    def test_train_before_test(self):
        folds = generate_folds(n_samples=5000, n_folds=5)
        for fold in folds:
            assert fold.train_end <= fold.test_start, (
                f"Fold {fold.fold_id}: train_end={fold.train_end} > test_start={fold.test_start}"
            )

    def test_folds_are_expanding(self):
        folds = generate_folds(n_samples=5000, n_folds=5)
        for i in range(1, len(folds)):
            assert folds[i].train_end > folds[i - 1].train_end, (
                "Training window must expand with each fold"
            )


class TestPurgeGap:
    """§5.4 — Gap of at least L + max(h) between train and test."""

    def test_purge_gap_exists(self):
        folds = generate_folds(n_samples=5000, n_folds=5, input_window=168, max_horizon=24)
        min_gap = 168 + 24  # L + max(h)
        for fold in folds:
            actual_gap = fold.test_start - fold.train_end
            assert actual_gap >= min_gap, (
                f"Fold {fold.fold_id}: gap={actual_gap} < required={min_gap}"
            )

    def test_purge_gap_size_recorded(self):
        folds = generate_folds(n_samples=5000, n_folds=5)
        for fold in folds:
            assert fold.purge_size > 0


class TestWindowBoundaries:
    """§5.3 — Windows must not straddle split boundaries."""

    def test_no_straddling_windows(self):
        n = 5000
        folds = generate_folds(n_samples=n, n_folds=5, input_window=168, max_horizon=24)
        for fold in folds:
            # A window of size 168 ending at train_end should not
            # have its target reaching into the purge zone or test set
            last_valid_window_end = fold.train_end
            first_test_target = fold.test_start
            gap = first_test_target - last_valid_window_end
            assert gap >= 168 + 24, (
                f"Fold {fold.fold_id}: boundary window could straddle. "
                f"Gap={gap}, need >= {168 + 24}"
            )


class TestScalerLeakage:
    """§5.2 — The backtest fits its scaler on each fold's training span only."""

    def test_backtest_scaler_fit_on_train_only(self):
        import pandas as pd

        from forecast.evaluate.backtest import run_backtest
        from forecast.models.naive import NaiveForecaster

        n = 4000
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        # Level jumps from 10 to 1000 half-way: a scaler fit on all data would see it.
        values = np.where(np.arange(n) < n // 2, 10.0, 1000.0) + np.sin(np.arange(n))
        series = pd.Series(values, index=idx)
        result = run_backtest(
            NaiveForecaster,
            series,
            input_window=168,
            horizons=[1],
            seasonal_period=24,
            n_folds=2,
        )
        for info in result.folds:
            train = values[info.fold.train_start : info.fold.train_end]
            assert info.scaler_mean == pytest.approx(train.mean())
            assert info.scaler_scale == pytest.approx(train.std())
