"""Point and interval metrics. MASE (vs seasonal naive on the training data) is the headline."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """MAPE in percent, or ``None`` when any true value is near zero."""
    if np.any(np.abs(y_true) < 1e-6):
        return None
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mase_scale(y_train: np.ndarray, seasonal_period: int) -> float:
    """In-sample MAE of the seasonal-naive forecast: the MASE denominator."""
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = float(np.mean(naive_errors))
    if scale < 1e-10:
        raise ValueError(
            "MASE denominator near zero - training series has no seasonal variation. "
            "Check the seasonal period or data."
        )
    return scale


def mase_from_scale(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int,
) -> float:
    """Mean Absolute Scaled Error; < 1 beats in-sample seasonal naive."""
    return mase_from_scale(y_true, y_pred, mase_scale(y_train, seasonal_period))


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


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals inside ``[lower, upper]``."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))
