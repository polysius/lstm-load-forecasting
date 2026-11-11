"""Rolling-origin evaluation of a served artefact on a fixed time window.

Used for the candidate's holdout metrics and by the promotion rule, which scores
the candidate and the current production bundle on the *same* window with the
*same* MASE denominator, against baselines computed on the same windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast.evaluate.metrics import interval_coverage, interval_width


@dataclass(frozen=True)
class Windows:
    origins: np.ndarray  # position of the last observed point
    X: np.ndarray  # (n, L) inputs, raw units
    Y: np.ndarray  # (n, H) targets, raw units
    hours: np.ndarray  # (n, L) epoch hours of the inputs


def rolling_windows(
    values: np.ndarray,
    epoch_hours: np.ndarray,
    start: int,
    end: int,
    input_window: int,
    horizon: int,
    stride: int = 1,
) -> Windows:
    """Every origin whose ``horizon`` targets lie in ``[start, end)``; inputs may precede it."""
    origins = np.arange(start - 1, end - horizon, stride)
    origins = origins[origins - input_window + 1 >= 0]
    in_idx = origins[:, None] + np.arange(-input_window + 1, 1)[None, :]
    out_idx = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    X, Y = values[in_idx], values[out_idx]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return Windows(origins[ok], X[ok], Y[ok], epoch_hours[in_idx][ok])


def seasonal_scale(values: np.ndarray, period: int = 24) -> float:
    """In-sample seasonal-naive MAE (NaN-safe): the MASE denominator."""
    diffs = np.abs(values[period:] - values[:-period])
    scale = float(np.nanmean(diffs))
    if not np.isfinite(scale) or scale < 1e-10:
        raise ValueError("cannot compute a MASE scale from the reference span")
    return scale


def baseline_predictions(X: np.ndarray, horizon: int, kind: str, period: int = 24) -> np.ndarray:
    if kind == "naive":
        return np.repeat(X[:, -1:], horizon, axis=1)
    if kind == "seasonal_naive":
        return X[:, -period:][:, np.arange(horizon) % period]
    raise ValueError(kind)


def point_metrics(
    Y: np.ndarray, P: np.ndarray, scale: float, horizons: tuple[int, ...]
) -> dict[str, float]:
    """Errors pooled over steps 1..h (as in the backtest) for each reported horizon."""
    out: dict[str, float] = {}
    for h in horizons:
        err = Y[:, :h] - P[:, :h]
        out[f"mae_h{h}"] = float(np.mean(np.abs(err)))
        out[f"rmse_h{h}"] = float(np.sqrt(np.mean(err**2)))
        out[f"mase_h{h}"] = out[f"mae_h{h}"] / scale
    out["mase_mean"] = float(np.mean([out[f"mase_h{h}"] for h in horizons]))
    out["mae_mean"] = float(np.mean([out[f"mae_h{h}"] for h in horizons]))
    return out


def evaluate_predictor(
    predictor,
    values: np.ndarray,
    epoch_hours: np.ndarray,
    start: int,
    end: int,
    *,
    horizons: tuple[int, ...],
    scale: float,
    stride: int = 1,
) -> tuple[dict[str, float], Windows, np.ndarray]:
    """Point and interval metrics of a Predictor on ``[start, end)``; also the windows."""
    H = predictor.max_horizon
    w = rolling_windows(values, epoch_hours, start, end, predictor.input_window, H, stride)
    if len(w.origins) == 0:
        raise ValueError("no complete evaluation window in the requested span")
    P = predictor.predict_windows(w.X, w.hours)
    metrics = point_metrics(w.Y, P, scale, horizons)
    for level in predictor.levels:
        _, half = predictor.interval_halfwidth(level, H)
        lower, upper = np.maximum(P - half, 0.0), P + half
        metrics[f"coverage_{level}"] = interval_coverage(w.Y, lower, upper)
        metrics[f"width_{level}"] = interval_width(lower, upper)
    metrics["n_origins"] = float(len(w.origins))
    return metrics, w, P


def baseline_metrics(
    w: Windows, scale: float, horizons: tuple[int, ...], horizon: int
) -> dict[str, dict[str, float]]:
    return {
        kind: point_metrics(w.Y, baseline_predictions(w.X, horizon, kind), scale, horizons)
        for kind in ("naive", "seasonal_naive")
    }
