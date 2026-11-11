"""Conformal half-widths, rolling holdout windows and pooled metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from forecast.evaluate.holdout import (
    baseline_predictions,
    point_metrics,
    rolling_windows,
    seasonal_scale,
)
from forecast.features import calendar_np
from forecast.train.conformal import conformal_halfwidths


class TestConformal:
    def test_rank_is_ceil_n_plus_one_times_level(self):
        res = np.arange(1, 101, dtype=float)[:, None]  # 1..100
        q = conformal_halfwidths(res, [0.8, 0.9])
        assert q["0.8"] == [81.0]  # ceil(101 * 0.8) = 81st smallest
        assert q["0.9"] == [91.0]

    def test_empirical_coverage_on_exchangeable_residuals(self):
        rng = np.random.default_rng(0)
        calib, test = np.abs(rng.normal(size=(2000, 3))), np.abs(rng.normal(size=(20000, 3)))
        for level in (0.8, 0.95):
            q = np.asarray(conformal_halfwidths(calib, [level])[f"{level:g}"])
            coverage = (test <= q).mean(axis=0)
            assert np.all(np.abs(coverage - level) < 0.02)

    def test_tiny_calibration_set_clamps_to_max(self):
        q = conformal_halfwidths(np.array([[1.0], [2.0]]), [0.95])
        assert q["0.95"] == [2.0]

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
    def test_invalid_levels(self, bad):
        with pytest.raises(ValueError):
            conformal_halfwidths(np.ones((5, 2)), [bad])


class TestRollingWindows:
    def test_targets_stay_inside_the_span_and_inputs_end_at_origin(self):
        values = np.arange(500, dtype=float)
        hours = np.arange(500)
        w = rolling_windows(values, hours, start=300, end=400, input_window=168, horizon=24)
        assert w.Y.min() == 300 and w.Y.max() == 399
        np.testing.assert_array_equal(w.X[:, -1], w.origins)
        np.testing.assert_array_equal(w.Y[:, 0], w.origins + 1)
        assert len(w.origins) == 100 - 24 + 1

    def test_windows_touching_nan_are_dropped(self):
        values = np.arange(500, dtype=float)
        values[350] = np.nan
        w = rolling_windows(values, np.arange(500), 300, 400, 168, 24)
        assert not np.isnan(w.X).any() and not np.isnan(w.Y).any()


class TestMetrics:
    def test_errors_pool_steps_one_to_h(self):
        Y = np.zeros((2, 24))
        P = np.zeros((2, 24))
        P[:, 23] = 24.0  # only step 24 is wrong
        m = point_metrics(Y, P, scale=1.0, horizons=(1, 6, 24))
        assert m["mae_h1"] == 0 and m["mae_h6"] == 0 and m["mae_h24"] == pytest.approx(1.0)

    def test_baselines(self):
        X = np.arange(168, dtype=float)[None, :]
        np.testing.assert_array_equal(baseline_predictions(X, 3, "naive"), [[167, 167, 167]])
        np.testing.assert_array_equal(
            baseline_predictions(X, 3, "seasonal_naive"), [[144, 145, 146]]
        )

    def test_seasonal_scale_ignores_nan(self):
        v = np.tile(np.arange(24.0), 10) + np.repeat(np.arange(10.0), 24)
        v[5] = np.nan
        assert seasonal_scale(v) == pytest.approx(1.0)


class TestCalendarNumpy:
    def test_matches_pandas(self):
        import pandas as pd

        idx = pd.date_range("2025-03-28", periods=24 * 9, freq="h", tz="UTC")
        enc = calendar_np.encode(calendar_np.epoch_hours(idx), calendar_np.AVAILABLE)
        np.testing.assert_allclose(enc[:, 0], np.sin(2 * np.pi * idx.hour / 24), atol=1e-12)
        np.testing.assert_allclose(enc[:, 2], np.sin(2 * np.pi * idx.day_of_week / 7), atol=1e-12)

    def test_hourly_range_ends_at_last(self):
        from datetime import UTC, datetime

        last = datetime(2025, 1, 1, 5, tzinfo=UTC)
        h = calendar_np.hourly_range(last, 3)
        assert list(h % 24) == [3, 4, 5]
        assert math.isclose(h[-1] / 24, (last - datetime(1970, 1, 1, tzinfo=UTC)).days + 5 / 24)
