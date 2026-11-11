"""SARIMA baseline (the repo's ARIMA-family baseline, kept as in the original study)."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class SARIMAForecaster:
    """SARIMAX(p,d,q)(P,D,Q,s) fit on the most recent ``max_train_periods`` seasons.

    At predict time the fitted parameters are re-applied to the last
    ``apply_periods`` seasons of the input window (no refit), then forecast.
    """

    name = "sarima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
        max_train_periods: int = 30,
        apply_periods: int = 2,
        maxiter: int = 50,
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.max_train_periods = max_train_periods
        self.apply_periods = apply_periods
        self.maxiter = maxiter
        self._fit_result = None

    @property
    def _period(self) -> int:
        return int(self.seasonal_order[3])

    def fit(self, y_train: np.ndarray, **context: Any) -> SARIMAForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        max_train = self._period * self.max_train_periods
        y = y_train[-max_train:] if len(y_train) > max_train else y_train
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit_result = model.fit(disp=False, maxiter=self.maxiter)
        log.info("SARIMA fit on %d points in %.1fs", len(y), time.perf_counter() - start)
        return self

    def predict(self, y_input: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        if self._fit_result is None:
            return np.full(horizon, y_input[-1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                updated = self._fit_result.apply(y_input[-self._period * self.apply_periods :])
                return np.asarray(updated.forecast(steps=horizon))
        except Exception:  # numerical failure on a window: fall back to persistence
            return np.full(horizon, y_input[-1])
