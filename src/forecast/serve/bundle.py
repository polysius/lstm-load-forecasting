"""The model bundle: ``model.onnx`` + ``metadata.json`` (the model card).

The bundle is the unit that is trained, registered, promoted, baked into the
serving image and evaluated by the promotion rule - always the same artefact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MODEL_FILE = "model.onnx"
CARD_FILE = "metadata.json"


class ScalerParams(BaseModel):
    mean: float
    scale: float


class IntervalSpec(BaseModel):
    method: str = "split-conformal"
    # level (e.g. "0.8") -> per-step absolute-residual quantiles, in target units
    levels: dict[str, list[float]]
    default_level: str
    calibration: dict[str, Any] = Field(default_factory=dict)


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    schema_version: int = 1
    model_name: str
    model_version: str = "dev"
    run_id: str | None = None
    created_at: str
    architecture: dict[str, Any]
    input_window: int
    max_horizon: int
    freq: str = "1h"
    covariates: list[str] = Field(default_factory=list)
    units: str = ""
    series: dict[str, Any] = Field(default_factory=dict)
    scaler: ScalerParams
    intervals: IntervalSpec
    training: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


def write_card(card: ModelCard, bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / CARD_FILE
    path.write_text(json.dumps(card.model_dump(), indent=2) + "\n", encoding="utf-8")
    return path


def read_card(bundle_dir: Path) -> ModelCard:
    return ModelCard.model_validate_json((bundle_dir / CARD_FILE).read_text(encoding="utf-8"))
