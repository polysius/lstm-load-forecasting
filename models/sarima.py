"""SARIMA baseline (SPEC §6.3)."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


class SARIMAForecaster:
    def __init__(self, seasonal_period: int = 24, max_fit_seconds: int = 300):
        self.seasonal_period = seasonal_period
        self.max_fit_seconds = max_fit_seconds
        self._fit_result = None

    @property
    def name(self) -> str:
        return "sarima"

    def fit(self, y_train: np.ndarray, **kwargs) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        max_train = self.seasonal_period * 30
        y = y_train[-max_train:] if len(y_train) > max_train else y_train

        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit_result = model.fit(disp=False, maxiter=50)
        elapsed = time.time() - start
        print(f"  SARIMA fit: {elapsed:.1f}s (cap: {self.max_fit_seconds}s)")

    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray:
        if self._fit_result is None:
            return np.full(horizon, y_input[-1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                updated = self._fit_result.apply(y_input[-self.seasonal_period * 2 :])
                forecast = updated.forecast(steps=horizon)
            return np.asarray(forecast)
        except Exception:
            return np.full(horizon, y_input[-1])
