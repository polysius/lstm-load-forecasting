"""Inference on a model bundle with ONNX Runtime. NumPy only - no torch, no pandas."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from forecast.features import calendar_np
from forecast.serve.bundle import MODEL_FILE, ModelCard, read_card


class InputError(ValueError):
    """The request cannot be forecast (too short, non-finite, bad horizon or level)."""


@dataclass(frozen=True)
class Forecast:
    timestamps: list[datetime]
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: str


class Predictor:
    def __init__(self, card: ModelCard, session) -> None:
        self.card = card
        self.session = session
        self._input_name = session.get_inputs()[0].name
        self._quantiles = {
            k: np.asarray(v, dtype=np.float64) for k, v in card.intervals.levels.items()
        }

    @classmethod
    def from_dir(cls, bundle_dir: str | Path, *, threads: int | None = None) -> Predictor:
        import onnxruntime as ort

        bundle_dir = Path(bundle_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or int(os.environ.get("ORT_THREADS", "0") or 0)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(bundle_dir / MODEL_FILE), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        return cls(read_card(bundle_dir), session)

    # ------------------------------------------------------------------ properties

    @property
    def levels(self) -> list[str]:
        return sorted(self._quantiles)

    @property
    def input_window(self) -> int:
        return self.card.input_window

    @property
    def max_horizon(self) -> int:
        return self.card.max_horizon

    # ------------------------------------------------------------------ core

    def _features(self, scaled: np.ndarray, epoch_hours: np.ndarray | None) -> np.ndarray:
        """``(n, L)`` scaled windows (+ ``(n, L)`` epoch hours) -> ``(n, L, 1 + k)`` float32."""
        x = scaled[:, :, None]
        if self.card.covariates:
            if epoch_hours is None:
                raise InputError("this model needs timestamps for its calendar covariates")
            cov = calendar_np.encode(epoch_hours.ravel(), self.card.covariates)
            x = np.concatenate([x, cov.reshape(*epoch_hours.shape, -1)], axis=-1)
        return np.ascontiguousarray(x, dtype=np.float32)

    def predict_windows(
        self, windows: np.ndarray, epoch_hours: np.ndarray | None = None
    ) -> np.ndarray:
        """Raw-unit windows ``(n, L)`` -> raw-unit point forecasts ``(n, max_horizon)``."""
        windows = np.asarray(windows, dtype=np.float64)
        s = self.card.scaler
        scaled = (windows - s.mean) / s.scale
        out = self.session.run(None, {self._input_name: self._features(scaled, epoch_hours)})[0]
        return out.astype(np.float64) * s.scale + s.mean

    def interval_halfwidth(self, level: str | None, horizon: int) -> tuple[str, np.ndarray]:
        level = level or self.card.intervals.default_level
        if level not in self._quantiles:
            raise InputError(f"coverage level {level!r} not calibrated; choose from {self.levels}")
        return level, self._quantiles[level][:horizon]

    def predict(
        self,
        history: list[float] | np.ndarray,
        last_timestamp: datetime,
        horizon: int,
        level: str | None = None,
    ) -> Forecast:
        history = np.asarray(history, dtype=np.float64)
        if not 1 <= horizon <= self.max_horizon:
            raise InputError(f"horizon must be in 1..{self.max_horizon}, got {horizon}")
        if len(history) < self.input_window:
            raise InputError(f"need at least {self.input_window} hourly values, got {len(history)}")
        window = history[-self.input_window :]
        if not np.all(np.isfinite(window)):
            raise InputError("history contains NaN or infinite values in the input window")
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=UTC)
        hours = calendar_np.hourly_range(last_timestamp, self.input_window)[None, :]
        point = self.predict_windows(window[None, :], hours)[0, :horizon]
        level, half = self.interval_halfwidth(level, horizon)
        stamps = [
            last_timestamp.astimezone(UTC) + timedelta(hours=k) for k in range(1, horizon + 1)
        ]
        # Load is non-negative; clipping the lower bound never reduces coverage.
        return Forecast(stamps, point, np.maximum(point - half, 0.0), point + half, level)
