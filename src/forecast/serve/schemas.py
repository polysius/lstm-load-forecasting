"""Request/response models: these *are* the API contract (see tests/contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]

EXAMPLE_HISTORY = [round(90000 + 15000 * ((h % 24) / 23), 1) for h in range(168)]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "series_id": "PJM",
                    "history": EXAMPLE_HISTORY,
                    "last_timestamp": "2026-11-09T23:00:00Z",
                    "horizon": 24,
                    "level": "0.8",
                }
            ]
        },
    )

    series_id: str | None = Field(
        None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Free-form series label, echoed back and logged.",
    )
    history: list[NonNegativeFinite] = Field(
        ...,
        min_length=1,
        max_length=8760,
        description="Hourly load, oldest first; the model uses the last `input_window` values.",
    )
    last_timestamp: AwareDatetime = Field(
        ..., description="Timestamp (timezone-aware) of the last history value."
    )
    horizon: int = Field(24, ge=1, le=168, description="Hours ahead to forecast.")
    level: str | None = Field(
        None, description="Interval coverage level, e.g. '0.8'; defaults to the model's default."
    )


class ForecastPoint(BaseModel):
    timestamp: datetime
    step: int
    point: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    series_id: str | None
    origin: datetime
    level: str
    units: str
    forecasts: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_version: str
    input_window: int
    max_horizon: int
    levels: list[str]
    uptime_seconds: float


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    detail: str | list[ValidationIssue]
