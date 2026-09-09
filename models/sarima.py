"""SARIMA baseline (SPEC §6.3)."""

from __future__ import annotations

import time
import warnings

import numpy as np
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


class SARIMAForecaster:
    def __init__(self, seasonal_period: int = 24, max_fit_seconds: int = 300):
        self.seasonal_period = seasonal_period
        self.max_fit_seconds = max_fit_seconds
        self._model = None

    @property
    def name(self) -> str:
        return "sarima"

    def fit(self, y_train: np.ndarray, **kwargs) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Use a subset for fitting if series is very long (speed cap)
        max_train = self.seasonal_period * 60
        y = y_train[-max_train:] if len(y_train) > max_train else y_train

        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit_result = self._model.fit(disp=False, maxiter=50)
        elapsed = time.time() - start
        print(f"  SARIMA fit: {elapsed:.1f}s (cap: {self.max_fit_seconds}s)")

    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray:
        if self._fit_result is None:
            return np.full(horizon, np.nan)
        try:
            forecast = self._fit_result.forecast(steps=horizon)
            return np.asarray(forecast)
        except Exception:
            return np.full(horizon, y_input[-1])
