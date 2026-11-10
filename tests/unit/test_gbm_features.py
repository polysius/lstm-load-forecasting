"""Train/serve feature parity for the GBM: the bug class that skewed the original baseline.

For every step ``s`` the feature row built at *training* time for target index ``T``
must equal the row built at *predict* time from the window ending at the forecast
origin ``T - s``. The ``target`` alignment satisfies this; ``legacy`` provably does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.models.gbm import GBMForecaster

N = 800
WINDOW = 168


@pytest.fixture
def series() -> tuple[np.ndarray, pd.DatetimeIndex]:
    rng = np.random.default_rng(0)
    y = rng.normal(size=N)
    ts = pd.date_range("2024-03-01 05:00", periods=N, freq="h", tz="UTC")
    return y, ts


@pytest.mark.parametrize("step", [1, 2, 6, 13, 24])
def test_target_alignment_train_serve_parity(series, step):
    y, ts = series
    gbm = GBMForecaster(feature_alignment="target")
    X_train, targets = gbm._target_train_features(y, ts, step, None)
    target_idx = np.arange(max(gbm._valid_lags(step)), N)
    for row in (0, len(target_idx) // 2, len(target_idx) - 1):
        T = target_idx[row]
        origin = T - step
        if origin - WINDOW + 1 < 0:
            continue
        window = y[origin - WINDOW + 1 : origin + 1][None, :]
        X_pred = gbm._target_predict_features(window, step, ts[[origin]], None)
        np.testing.assert_array_equal(X_train[row], X_pred[0])
        assert targets[row] == y[T]


def test_target_lags_are_relative_to_the_target(series):
    y = np.arange(N, dtype=float)  # y[t] == t makes every feature its own index
    ts = series[1]
    gbm = GBMForecaster(feature_alignment="target")
    X, targets = gbm._target_train_features(y, ts, 24, None)
    lags = gbm._valid_lags(24)  # [24, 48, 168]
    np.testing.assert_array_equal(X[:, : len(lags)], targets[:, None] - np.array(lags))


def test_legacy_alignment_has_train_serve_skew(series):
    """Documents the original bug: predict-time calendar features are constant."""
    y, _ = series
    gbm = GBMForecaster(feature_alignment="legacy")
    X_train, _ = gbm._legacy_train_features(y, 1)
    assert len(np.unique(X_train[:, -2])) == 24  # training sees every hour...
    windows = np.stack([y[i : i + WINDOW] for i in range(0, 200, 7)])
    X_pred = gbm._legacy_predict_features(windows, 1)
    assert np.all(X_pred[:, -2:] == 0)  # ...serving always says "hour 0, day 0"


def test_legacy_lags_are_misaligned_for_multi_step(series):
    y = np.arange(N, dtype=float)
    gbm = GBMForecaster(feature_alignment="legacy")
    X, targets = gbm._legacy_train_features(y, 24)
    # The nominal 24h lag actually reads target - 47.
    np.testing.assert_array_equal(targets - X[:, 0], np.full(len(X), 47.0))


def test_fit_predict_roundtrip_target(series):
    y, ts = series
    gbm = GBMForecaster(feature_alignment="target", n_estimators=20, n_jobs=1, seed=0)
    gbm.fit(y, timestamps=ts, horizons=[6])
    windows = np.stack([y[i : i + WINDOW] for i in range(3)])
    preds = gbm.predict_batch(windows, 6, origin_times=ts[[WINDOW - 1, WINDOW, WINDOW + 1]])
    assert preds.shape == (3, 6) and np.isfinite(preds).all()
    single = gbm.predict(windows[0], 6, origin_time=ts[WINDOW - 1])
    np.testing.assert_allclose(single, preds[0])
