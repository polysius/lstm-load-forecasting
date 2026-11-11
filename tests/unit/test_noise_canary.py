"""Noise canaries: if the pipeline leaks, models gain skill they cannot legitimately have.

1. Shuffled future-target canary (spec M1.3). A covariate holding y(t+1), randomly
   permuted in time, carries no usable information; training with it must not
   improve the backtest. The *unshuffled* covariate is a positive control: it is a
   genuine leak, and the harness must show a large gain - otherwise the negative
   result would prove nothing.
2. Pure-noise floor. On i.i.d. noise the best possible forecast is the mean, whose
   MASE is 1/sqrt(2) ~= 0.707 (the seasonal-naive denominator on noise is sqrt(2)
   times the mean's MAE). Scoring clearly below that floor requires leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.evaluate.backtest import run_backtest, score_forecasts
from forecast.models.gbm import GBMForecaster
from forecast.models.naive import NaiveForecaster, SeasonalNaiveForecaster

N = 6000
INPUT_WINDOW = 168
NOISE_FLOOR = 1 / np.sqrt(2)


def seasonal_series(n: int = N, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = np.zeros(n)
    eps = rng.normal(0, 3.0, n)
    for i in range(1, n):  # AR(1) noise so persistence has something to exploit
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    y = 100 + 15 * np.sin(2 * np.pi * t / 24) + 6 * np.sin(2 * np.pi * t / 168) + noise
    idx = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    return pd.Series(y, index=idx, name="load")


def mase_by_horizon(factory, series, *, horizons, covariates=None, n_folds=2) -> pd.Series:
    result = run_backtest(
        factory,
        series,
        input_window=INPUT_WINDOW,
        horizons=horizons,
        seasonal_period=24,
        n_folds=n_folds,
        covariates=covariates,
    )
    return score_forecasts(result).groupby("horizon")["mase"].mean()


def future_target(series: pd.Series) -> np.ndarray:
    """y(t+1) aligned at t (last value carried forward)."""
    return series.shift(-1).ffill().to_numpy()


def small_gbm() -> GBMForecaster:
    return GBMForecaster(feature_alignment="target", n_estimators=120, n_jobs=1, seed=0)


def small_lstm():
    from forecast.models.lstm import LSTMForecaster

    return LSTMForecaster(
        input_window=INPUT_WINDOW,
        max_horizon=6,
        hidden_size=16,
        max_epochs=8,
        patience=3,
        batch_size=128,
        seed=0,
    )


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return seasonal_series()


@pytest.fixture(scope="module")
def noise() -> pd.Series:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-02", periods=N, freq="h", tz="UTC")
    return pd.Series(rng.standard_normal(N), index=idx, name="load")


class TestShuffledFutureTargetCanary:
    def _canary(self, series: pd.Series, *, shuffled: bool) -> pd.DataFrame:
        values = future_target(series)
        if shuffled:
            values = np.random.default_rng(1).permutation(values)
        return pd.DataFrame({"canary": values}, index=series.index)

    @pytest.mark.parametrize(
        "factory", [small_gbm, pytest.param(small_lstm, marks=pytest.mark.slow)]
    )
    def test_shuffled_future_target_does_not_help(self, series, factory):
        base = mase_by_horizon(factory, series, horizons=(1, 6))
        canary = mase_by_horizon(
            factory, series, horizons=(1, 6), covariates=self._canary(series, shuffled=True)
        )
        # No horizon may improve by more than 3% just from a column of shuffled noise.
        assert (canary >= 0.97 * base).all(), f"shuffled canary helped:\n{base}\n{canary}"

    @pytest.mark.parametrize(
        ("factory", "max_ratio"),
        [
            (small_gbm, 0.5),
            # The tiny LSTM exploits the leaked covariate less sharply than trees do
            # (measured leaky/base = 0.72 on CPU), but a shuffled canary stays near 1.0.
            pytest.param(small_lstm, 0.85, marks=pytest.mark.slow),
        ],
    )
    def test_unshuffled_future_target_is_detected(self, series, factory, max_ratio):
        """Positive control: a real leak must produce a clearly visible gain."""
        base = mase_by_horizon(factory, series, horizons=(1, 6))
        leaky = mase_by_horizon(
            factory, series, horizons=(1, 6), covariates=self._canary(series, shuffled=False)
        )
        assert leaky[1] < max_ratio * base[1], (
            f"leak not detected: base={base[1]:.3f} leaky={leaky[1]:.3f}"
        )


class TestPureNoiseFloor:
    @pytest.mark.parametrize(
        "factory",
        [
            NaiveForecaster,
            SeasonalNaiveForecaster,
            small_gbm,
            lambda: GBMForecaster(feature_alignment="legacy", n_estimators=120, n_jobs=1),
            pytest.param(small_lstm, marks=pytest.mark.slow),
        ],
        ids=["naive", "seasonal_naive", "gbm", "gbm_legacy", "lstm"],
    )
    def test_no_model_beats_the_noise_floor(self, noise, factory):
        mase = mase_by_horizon(factory, noise, horizons=(1, 6))
        assert (mase > NOISE_FLOOR - 0.05).all(), f"skill on pure noise => leakage:\n{mase}"
