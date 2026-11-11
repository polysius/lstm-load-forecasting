"""Service metrics: Prometheus for scraping, CloudWatch EMF lines for Lambda."""

from __future__ import annotations

import json
import os
import sys
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class ServiceMetrics:
    """One registry per app instance, so tests can build many apps side by side."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "forecast_requests_total",
            "HTTP requests",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "forecast_request_latency_seconds",
            "End-to-end latency",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inference = Histogram(
            "forecast_inference_latency_seconds",
            "ONNX Runtime inference latency",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.log_errors = Counter(
            "forecast_prediction_log_errors_total",
            "Prediction log writes that failed",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "forecast_model_info",
            "Loaded model",
            ["model_name", "model_version"],
            registry=self.registry,
        )


def emit_emf(metrics: dict[str, float], *, service: str, model_version: str) -> None:
    """Print a CloudWatch Embedded Metric Format record (Lambda turns it into metrics).

    Enabled with ``METRICS_EMF=1``; costs nothing off Lambda because nobody parses it.
    """
    if os.environ.get("METRICS_EMF") != "1":
        return
    namespace = os.environ.get("METRICS_NAMESPACE", "LoadForecaster")
    record = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [["Service"]],
                    "Metrics": [{"Name": k, "Unit": "Milliseconds"} for k in metrics],
                }
            ],
        },
        "Service": service,
        "ModelVersion": model_version,
        **metrics,
    }
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()
