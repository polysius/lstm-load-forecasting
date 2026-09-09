"""Tests for evaluation metrics, especially MASE edge cases (SPEC §8.2)."""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import compute_all_metrics, mae, mape, mase, rmse


class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == 1.0


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_ge_mae(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestMAPE:
    def test_returns_none_for_near_zero(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 1.1, 2.1])
        assert mape(y_true, y_pred) is None

    def test_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        result = mape(y_true, y_pred)
        assert result is not None
        assert abs(result - 10.0) < 0.01


class TestMASE:
    def test_perfect_prediction(self):
        rng = np.random.default_rng(42)
        pattern = np.tile([10.0, 20.0, 30.0, 20.0], 50)
        y_train = pattern + rng.normal(0, 0.5, size=len(pattern))
        y_true = np.array([10.0, 20.0, 30.0])
        result = mase(y_true, y_true, y_train, seasonal_period=4)
        assert result == 0.0

    def test_mase_equals_one_for_seasonal_naive(self):
        rng = np.random.default_rng(42)
        seasonal_period = 24
        n_train = 500
        n_test = 48

        pattern = rng.uniform(5, 15, size=seasonal_period)
        y_train = np.tile(pattern, n_train // seasonal_period + 1)[:n_train]
        noise = rng.normal(0, 0.1, size=len(y_train))
        y_train = y_train + noise

        y_true = y_train[-n_test:]
        y_pred_seasonal = y_train[-n_test - seasonal_period : -seasonal_period]

        result = mase(y_true, y_pred_seasonal, y_train, seasonal_period)
        assert abs(result - 1.0) < 0.5  # approximately 1 with noise

    def test_zero_denominator_raises(self):
        y_train = np.ones(100)  # constant series => zero seasonal variation
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 2.5])
        with pytest.raises(ValueError, match="denominator near zero"):
            mase(y_true, y_pred, y_train, seasonal_period=24)

    def test_mase_below_one_means_beats_baseline(self):
        seasonal_period = 24
        y_train = np.sin(np.linspace(0, 10 * np.pi, 500)) * 10 + 50
        y_true = np.sin(np.linspace(10 * np.pi, 11 * np.pi, 48)) * 10 + 50
        # Near-perfect prediction
        y_pred = y_true + np.random.default_rng(42).normal(0, 0.01, size=48)
        result = mase(y_true, y_pred, y_train, seasonal_period)
        assert result < 1.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        rng = np.random.default_rng(42)
        y_train = rng.uniform(10, 100, size=500)
        y_true = rng.uniform(10, 100, size=50)
        y_pred = y_true + rng.normal(0, 1, size=50)
        result = compute_all_metrics(y_true, y_pred, y_train, seasonal_period=24)
        assert set(result.keys()) == {"mae", "rmse", "mape", "mase"}
