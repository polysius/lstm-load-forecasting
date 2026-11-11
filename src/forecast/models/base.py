"""The interface every forecaster implements, so the backtest treats them identically."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """Direct multi-horizon forecaster over a (scaled) univariate series.

    ``fit`` receives the training slice of one fold. Optional keyword context the
    backtest passes and models may use or ignore:

    * ``timestamps`` - DatetimeIndex aligned with ``y_train``
    * ``covariates`` - ``(n_train, k)`` exogenous features aligned with ``y_train``
    * ``horizons``   - horizons that will be requested
    """

    name: str

    def fit(self, y_train: np.ndarray, **context: Any) -> Forecaster: ...

    def predict(self, y_input: np.ndarray, horizon: int, **context: Any) -> np.ndarray: ...


def predict_many(
    model: Forecaster, windows: np.ndarray, horizon: int, **context: Any
) -> np.ndarray:
    """``(n, L)`` input windows -> ``(n, horizon)`` forecasts.

    Uses the model's vectorised ``predict_batch`` when it has one; otherwise loops.
    Per-window context arrays (``origin_times``, ``covariates``) are sliced for the loop.
    """
    batch = getattr(model, "predict_batch", None)
    if batch is not None:
        out = np.asarray(batch(windows, horizon, **context), dtype=float)
        return out[:, :horizon]
    rows = []
    for i, x in enumerate(windows):
        ctx = {k: (v[i] if v is not None else None) for k, v in context.items()}
        rows.append(np.asarray(model.predict(x, horizon=horizon, **ctx), dtype=float)[:horizon])
    return np.vstack(rows)
