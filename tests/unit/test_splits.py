"""Tests for rolling-origin splits (SPEC §8.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.evaluate.splits import generate_folds, split_data


class TestFoldGeneration:
    def test_minimum_five_folds(self):
        folds = generate_folds(n_samples=10000, n_folds=5)
        assert len(folds) >= 5

    def test_folds_cover_data(self):
        n = 10000
        folds = generate_folds(n_samples=n, n_folds=5)
        assert folds[0].train_start == 0
        assert folds[-1].test_end <= n

    def test_no_overlap_between_train_and_test(self):
        folds = generate_folds(n_samples=10000, n_folds=5)
        for fold in folds:
            assert fold.train_end < fold.test_start

    def test_train_expands_each_fold(self):
        folds = generate_folds(n_samples=10000, n_folds=5)
        for i in range(1, len(folds)):
            assert folds[i].train_end > folds[i - 1].train_end

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="Not enough data"):
            generate_folds(n_samples=100, n_folds=10, input_window=168, max_horizon=24)


class TestSplitData:
    def test_split_sizes_match_fold(self):
        n = 5000
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({"value": np.arange(n, dtype=float)}, index=idx)
        folds = generate_folds(n_samples=n, n_folds=3)

        for fold in folds:
            train, test = split_data(df, fold)
            assert len(train) == fold.train_end - fold.train_start
            assert len(test) == fold.test_end - fold.test_start

    def test_train_values_before_test_values(self):
        n = 5000
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({"value": np.arange(n, dtype=float)}, index=idx)
        folds = generate_folds(n_samples=n, n_folds=3)

        for fold in folds:
            train, test = split_data(df, fold)
            assert train.index[-1] < test.index[0]
