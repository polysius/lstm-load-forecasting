"""Seasonal naive baseline: y_hat(t+h) = y(t+h-s) (SPEC §6.2)."""

from __future__ import annotations

import numpy as np


class SeasonalNaiveForecaster:
    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = seasonal_period

    @property
    def name(self) -> str:
        return "seasonal_naive"

    def fit(self, y_train: np.ndarray, **kwargs) -> None:
        self._last_season = y_train[-self.seasonal_period :]

    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray:
        # Use the last full season from the input window
        season = y_input[-self.seasonal_period :]
        preds = np.empty(horizon)
        for i in range(horizon):
            preds[i] = season[i % self.seasonal_period]
        return preds
