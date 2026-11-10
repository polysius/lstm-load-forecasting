"""Model registry: build any forecaster from a name plus layered parameters."""

from __future__ import annotations

from typing import Any

from forecast.config import ModelName, SeasonalConfig, TaskConfig, load_model_defaults
from forecast.models.base import Forecaster, predict_many

__all__ = ["Forecaster", "build_model", "predict_many", "resolve_params"]


def resolve_params(model: ModelName, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """``configs/model/<model>.yaml`` defaults, then per-run overrides."""
    return {**load_model_defaults(model), **(overrides or {})}


def build_model(
    model: ModelName,
    params: dict[str, Any] | None = None,
    *,
    task: TaskConfig,
    seasonal: SeasonalConfig,
    seed: int | None = None,
) -> Forecaster:
    p = resolve_params(model, params)
    if model == "naive":
        from forecast.models.naive import NaiveForecaster

        return NaiveForecaster()
    if model == "seasonal_naive":
        from forecast.models.naive import SeasonalNaiveForecaster

        return SeasonalNaiveForecaster(
            seasonal_period=p.get("seasonal_period", seasonal.primary_period)
        )
    if model == "sarima":
        from forecast.models.sarima import SARIMAForecaster

        return SARIMAForecaster(**p)
    if model == "gbm":
        from forecast.models.gbm import GBMForecaster

        return GBMForecaster(**p, seed=seed)
    if model == "lstm":
        from forecast.models.lstm import LSTMForecaster

        return LSTMForecaster(
            input_window=task.input_window,
            max_horizon=task.max_horizon,
            seed=42 if seed is None else seed,
            **p,
        )
    raise ValueError(f"unknown model {model!r}")
