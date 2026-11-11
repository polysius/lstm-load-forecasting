"""Typed configuration loaded from the YAML files under ``configs/``.

Configs are validated with pydantic so a typo in a YAML key fails loudly at load
time instead of silently falling back to a default deep inside a training run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelName = Literal["naive", "seasonal_naive", "sarima", "gbm", "lstm"]


# --------------------------------------------------------------------------- paths


def project_root() -> Path:
    """Repository root: ``$FORECAST_ROOT``, else the checkout this file lives in."""
    env = os.environ.get("FORECAST_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parents[2]
    if (here / "configs").is_dir():
        return here
    return Path.cwd().resolve()


def configs_dir() -> Path:
    return project_root() / "configs"


def data_dir() -> Path:
    return Path(os.environ.get("FORECAST_DATA_DIR", project_root() / "data")).resolve()


def experiments_dir() -> Path:
    return Path(os.environ.get("FORECAST_EXPERIMENTS_DIR", project_root() / "experiments"))


# --------------------------------------------------------------------------- schema


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskConfig(_Strict):
    input_window: int = Field(168, gt=0, description="hours of history fed to the model")
    forecast_horizons: tuple[int, ...] = (1, 6, 24)

    @property
    def max_horizon(self) -> int:
        return max(self.forecast_horizons)


class SeasonalConfig(_Strict):
    primary_period: int = 24
    secondary_period: int = 168


class PreprocessingConfig(_Strict):
    resample_freq: str = "1h"
    timezone: str = "UTC"
    interpolate_method: str = "linear"
    interpolate_max_gap_hours: int = 3
    zero_is_missing: bool = True
    # Values further than this many robust z-scores (median/MAD) from the median
    # are treated as missing. ``None`` disables the filter (UCI reference setting).
    outlier_mad_threshold: float | None = None


class UCISource(_Strict):
    source_url: str
    raw_filename: str
    target_client: str


class EIASource(_Strict):
    base_url: str = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    respondent: str = "PJM"
    series_type: str = "D"
    history_days: int = 730
    page_size: int = Field(5000, le=5000)


class DataConfig(_Strict):
    name: str
    source: Literal["uci", "eia"]
    target_col: str = "load"
    units: str = ""
    uci: UCISource | None = None
    eia: EIASource | None = None
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    task: TaskConfig = TaskConfig()
    seasonal: SeasonalConfig = SeasonalConfig()

    @model_validator(mode="after")
    def _source_block_present(self) -> DataConfig:
        if getattr(self, self.source) is None:
            raise ValueError(f"source={self.source!r} requires a `{self.source}:` block")
        return self


class ModelSpec(_Strict):
    model: ModelName
    label: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    seeds: tuple[int, ...] | None = None

    @property
    def result_label(self) -> str:
        return self.label or self.model


class ReferenceSpec(_Strict):
    """A published results CSV to reproduce, with documented, evidenced exceptions."""

    path: str
    label: str = ""
    # model -> why its published numbers cannot be reproduced from the committed code.
    # Such rows are reported as KNOWN (never PASS) and do not fail the check.
    known_differences: dict[str, str] = Field(default_factory=dict)


class RunSpec(_Strict):
    name: str
    n_folds: int = Field(5, ge=1)
    models: tuple[ModelSpec, ...]
    references: tuple[ReferenceSpec, ...] = ()


class BenchmarkConfig(_Strict):
    name: str
    description: str = ""
    data: str
    runs: tuple[RunSpec, ...]
    tolerances: dict[str, float] = Field(default_factory=dict)


class PromotionRule(_Strict):
    """Champion/challenger gate; ``metric`` is lower-is-better on the shared holdout."""

    metric: str = "mase_mean"
    min_improvement: float = Field(0.0, ge=0.0)
    must_beat_baseline: Literal["naive", "seasonal_naive"] | None = "seasonal_naive"


class ProductionConfig(_Strict):
    model_name: str
    data: str = "eia"
    experiment: str = "production"
    seed: int = 42
    lstm: dict[str, Any] = Field(default_factory=dict)
    covariates: tuple[str, ...] = ()
    holdout_hours: int = Field(336, gt=0)
    calibration_hours: int = Field(336, gt=0)
    train_window_hours: int | None = None
    interval_levels: tuple[float, ...] = (0.8, 0.95)
    default_level: float = 0.8
    eval_stride: int = Field(1, ge=1)
    scale_reference_hours: int = Field(672, gt=24)
    promotion: PromotionRule = PromotionRule()

    @model_validator(mode="after")
    def _default_level_calibrated(self) -> ProductionConfig:
        if self.default_level not in self.interval_levels:
            raise ValueError("default_level must be one of interval_levels")
        return self


class MonitorConfig(_Strict):
    """Drift and rolling-error monitoring plus the retraining trigger (``configs/monitor.yaml``)."""

    series: str = "PJM"
    data: str = "eia"
    production: str = "production"
    window_hours: int = Field(168, gt=24)
    reference: Literal["year_over_year", "training_tail"] = "year_over_year"
    reference_hours: int = Field(672, gt=24)
    psi_bins: int = Field(10, ge=2)
    psi_threshold: float | Literal["auto"] = "auto"
    psi_threshold_quantile: float = Field(0.99, gt=0.5, lt=1.0)
    psi_threshold_floor: float = Field(0.2, gt=0.0)
    mase_threshold: float | None = None
    mase_degradation: float = Field(1.25, gt=1.0)
    consecutive_breaches: int = Field(2, ge=1)
    cooldown_hours: int = Field(72, ge=0)


# --------------------------------------------------------------------------- loaders


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config(name_or_path: str | Path, kind: str) -> Path:
    """``"uci"`` -> ``configs/data/uci.yaml``; explicit paths are used as-is."""
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"}:
        return p if p.is_absolute() else project_root() / p
    return configs_dir() / kind / f"{name_or_path}.yaml"


def load_data_config(name_or_path: str | Path) -> DataConfig:
    return DataConfig.model_validate(read_yaml(resolve_config(name_or_path, "data")))


def load_benchmark_config(name_or_path: str | Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(read_yaml(resolve_config(name_or_path, "train")))


def load_production_config(name_or_path: str | Path) -> ProductionConfig:
    return ProductionConfig.model_validate(read_yaml(resolve_config(name_or_path, "train")))


def load_monitor_config(name_or_path: str | Path = "monitor") -> MonitorConfig:
    path = resolve_config(name_or_path, ".")
    return MonitorConfig.model_validate(read_yaml(path))


def load_model_defaults(model: str) -> dict[str, Any]:
    """Default hyper-parameters from ``configs/model/<model>.yaml`` (empty if absent)."""
    path = configs_dir() / "model" / f"{model}.yaml"
    return read_yaml(path) if path.exists() else {}
