"""Persistence baselines: naive y(t+h) = y(t) and seasonal naive y(t+h) = y(t+h-s)."""

from __future__ import annotations

from typing import Any

import numpy as np


class NaiveForecaster:
    name = "naive"

    def fit(self, y_train: np.ndarray, **context: Any) -> NaiveForecaster:
        return self

    def predict(self, y_input: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        return np.full(horizon, y_input[-1])

    def predict_batch(self, windows: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        return np.repeat(windows[:, -1:], horizon, axis=1)


class SeasonalNaiveForecaster:
    name = "seasonal_naive"

    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = seasonal_period

    def fit(self, y_train: np.ndarray, **context: Any) -> SeasonalNaiveForecaster:
        return self

    def predict(self, y_input: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        season = y_input[-self.seasonal_period :]
        return season[np.arange(horizon) % self.seasonal_period].astype(float)

    def predict_batch(self, windows: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        season = windows[:, -self.seasonal_period :]
        return season[:, np.arange(horizon) % self.seasonal_period]
