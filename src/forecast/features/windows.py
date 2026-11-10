"""Supervised windows: the single place where inputs and targets are cut from a series."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_windows(
    y: np.ndarray,
    input_window: int,
    max_horizon: int,
    *,
    covariates: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``(X, Y)`` with ``X[i] = y[i : i+L]`` and ``Y[i] = y[i+L : i+L+H]``.

    ``X`` is ``(n, L, 1 + k)`` float32 when ``k`` covariates are given (covariate
    rows are taken over the *input* span only), ``(n, L, 1)`` otherwise.
    """
    y = np.asarray(y, dtype=np.float64)
    span = input_window + max_horizon
    if len(y) < span:
        raise ValueError(f"not enough data for windows: {len(y)} < {span}")
    views = sliding_window_view(y, span)[::stride]
    X = views[:, :input_window, None].astype(np.float32)
    Y = views[:, input_window:].astype(np.float32)
    if covariates is not None:
        cov = np.asarray(covariates, dtype=np.float64)
        if cov.ndim != 2 or len(cov) != len(y):
            raise ValueError(f"covariates must be (n, k) aligned with y; got {cov.shape}")
        cov_views = sliding_window_view(cov, span, axis=0)[::stride]  # (n, k, span)
        cov_in = np.transpose(cov_views[:, :, :input_window], (0, 2, 1)).astype(np.float32)
        X = np.concatenate([X, cov_in], axis=-1)
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)


def input_windows(series: np.ndarray, starts: np.ndarray, input_window: int) -> np.ndarray:
    """Stack ``series[s : s+L]`` for each start ``s`` into ``(n, L)``."""
    return (
        np.stack([series[s : s + input_window] for s in starts])
        if len(starts)
        else (np.empty((0, input_window)))
    )
