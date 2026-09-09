"""Naive persistence baseline: y_hat(t+h) = y(t) (SPEC §6.1)."""

from __future__ import annotations

import numpy as np


class NaiveForecaster:
    @property
    def name(self) -> str:
        return "naive"

    def fit(self, y_train: np.ndarray, **kwargs) -> None:
        pass

    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, y_input[-1])
