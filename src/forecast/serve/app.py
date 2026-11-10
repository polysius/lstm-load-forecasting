"""FastAPI forecasting service.

Runs unchanged under ``uvicorn`` locally, in Docker, and on AWS Lambda (the
Lambda Web Adapter extension forwards Function URL events to this HTTP server).

Environment:
  MODEL_DIR           bundle directory (default /opt/model)
  PREDICTION_LOG_URI  where to log predictions (see prediction_log.py)
  METRICS_EMF=1       also emit CloudWatch EMF metric lines
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from forecast import __version__
from forecast.serve.metrics import ServiceMetrics, emit_emf
from forecast.serve.prediction_log import PredictionLogger, from_uri
from forecast.serve.predictor import InputError, Predictor
from forecast.serve.schemas import (
    ErrorResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)

log = logging.getLogger("forecast.api")
SERVICE = "forecast-api"


def create_app(
    bundle_dir: str | None = None, prediction_logger: PredictionLogger | None = None
) -> FastAPI:
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = bundle_dir or os.environ.get("MODEL_DIR", "/opt/model")
        predictor = Predictor.from_dir(path)
        app.state.predictor = predictor
        app.state.prediction_log = prediction_logger or from_uri(
            os.environ.get("PREDICTION_LOG_URI")
        )
        app.state.started = time.monotonic()
        card = predictor.card
        metrics.model_info.labels(card.model_name, card.model_version).set(1)
        log.info(
            json.dumps(
                {
                    "event": "model_loaded",
                    "model_name": card.model_name,
                    "model_version": card.model_version,
                    "path": str(path),
                }
            )
        )
        yield

    app = FastAPI(
        title="Load Forecasting API",
        version=__version__,
        summary="Hourly electricity load forecasts with split-conformal prediction intervals.",
        lifespan=lifespan,
    )
    app.state.metrics = metrics

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Echo where and why, never the offending input: it may hold NaN (not JSON
        # serialisable) or a whole week of history.
        issues = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": issues})

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        metrics.requests.labels(endpoint, str(response.status_code)).inc()
        metrics.latency.labels(endpoint).observe(time.perf_counter() - start)
        return response

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        p: Predictor = request.app.state.predictor
        return HealthResponse(
            status="ok",
            model_name=p.card.model_name,
            model_version=p.card.model_version,
            input_window=p.input_window,
            max_horizon=p.max_horizon,
            levels=p.levels,
            uptime_seconds=round(time.monotonic() - request.app.state.started, 3),
        )

    @app.post(
        "/forecast",
        response_model=ForecastResponse,
        responses={422: {"model": ErrorResponse}},
        tags=["forecast"],
    )
    def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
        p: Predictor = request.app.state.predictor
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            fc = p.predict(body.history, body.last_timestamp, body.horizon, body.level)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        inference_s = time.perf_counter() - t0
        metrics.inference.observe(inference_s)

        origin = body.last_timestamp.astimezone(UTC)
        response = ForecastResponse(
            request_id=request_id,
            model_name=p.card.model_name,
            model_version=p.card.model_version,
            series_id=body.series_id,
            origin=origin,
            level=fc.level,
            units=p.card.units,
            forecasts=[
                ForecastPoint(
                    timestamp=ts,
                    step=k + 1,
                    point=float(fc.point[k]),
                    lower=float(fc.lower[k]),
                    upper=float(fc.upper[k]),
                )
                for k, ts in enumerate(fc.timestamps)
            ],
        )
        window = np.asarray(body.history[-p.input_window :], dtype=float)
        record = {
            "request_id": request_id,
            "logged_at": datetime.now(UTC).isoformat(),
            "model_name": p.card.model_name,
            "model_version": p.card.model_version,
            "series_id": body.series_id,
            "origin": origin.isoformat(),
            "level": fc.level,
            "inputs": window.round(3).tolist(),
            "forecasts": [f.model_dump(mode="json") for f in response.forecasts],
            "inference_ms": round(inference_s * 1000, 3),
        }
        try:
            request.app.state.prediction_log.log(record)
        except Exception:  # logging must never take the endpoint down
            metrics.log_errors.inc()
            log.exception("prediction log write failed")
        emit_emf(
            {"InferenceLatency": inference_s * 1000},
            service=SERVICE,
            model_version=p.card.model_version,
        )
        log.info(
            json.dumps(
                {
                    "event": "forecast",
                    "request_id": request_id,
                    "model_version": p.card.model_version,
                    "horizon": body.horizon,
                    "inference_ms": record["inference_ms"],
                }
            )
        )
        return response

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": SERVICE, "docs": "/docs", "health": "/health"}

    return app


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


_configure_logging()
app = create_app()
