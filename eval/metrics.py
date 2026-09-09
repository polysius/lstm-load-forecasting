"""Evaluation metrics including MASE (SPEC §8.2)."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Return MAPE only if no near-zero values in y_true. Otherwise return None."""
    if np.any(np.abs(y_true) < 1e-6):
        return None
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int,
) -> float:
    """Mean Absolute Scaled Error, scaled against seasonal naive on training data.

    MASE < 1 means the model beats seasonal naive. This is the headline metric (SPEC §8.2).
    """
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = np.mean(naive_errors)

    if scale < 1e-10:
        raise ValueError(
            "MASE denominator near zero — training series has no seasonal variation. "
            "Check the seasonal period or data."
        )

    forecast_errors = np.abs(y_true - y_pred)
    return float(np.mean(forecast_errors) / scale)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int,
) -> dict[str, float | None]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, seasonal_period),
    }
