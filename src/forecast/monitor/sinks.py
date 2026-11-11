"""Where monitoring metrics go: a Prometheus Pushgateway and a JSON lines file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

# metric name -> (Prometheus gauge name, help text)
METRICS = {
    "DriftPSI": ("forecast_drift_psi", "PSI of current vs reference load"),
    "DriftThreshold": ("forecast_drift_threshold", "Calibrated PSI alarm threshold"),
    "DriftBreach": ("forecast_drift_breach", "1 if PSI exceeds the threshold"),
    "DriftKS": ("forecast_drift_ks_statistic", "KS statistic, current vs reference"),
    "RollingMASE_h1": ("forecast_rolling_mase_h1", "7-day rolling MASE, step 1"),
    "RollingMASE_h24": ("forecast_rolling_mase_h24", "7-day rolling MASE, steps 1-24"),
    "Coverage80": ("forecast_interval_coverage", "7-day empirical interval coverage"),
    "RetrainTriggered": ("forecast_retrain_triggered", "1 if this check triggered"),
    "RetrainsTotal": ("forecast_retrains_total", "Retrains triggered so far"),
    "SimulatedDay": ("forecast_simulated_day", "Day index in the planted-shift demo"),
}


class MetricsSink(Protocol):
    def publish(self, values: dict[str, float], *, series: str, timestamp: datetime) -> None: ...


class PushgatewaySink:
    def __init__(self, url: str, job: str = "forecast-monitor"):
        self.url, self.job = url, job

    def publish(self, values: dict[str, float], *, series: str, timestamp: datetime) -> None:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        for key, value in values.items():
            if key not in METRICS or value is None:
                continue
            name, help_text = METRICS[key]
            Gauge(name, help_text, ["series"], registry=registry).labels(series).set(value)
        push_to_gateway(self.url, job=self.job, registry=registry)


class JsonLinesSink:
    def __init__(self, path: Path):
        self.path = Path(path)

    def publish(self, values: dict[str, float], *, series: str, timestamp: datetime) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"timestamp": timestamp.isoformat(), "series": series, **values}) + "\n"
            )


class FanOut:
    def __init__(self, *sinks: MetricsSink):
        self.sinks = sinks

    def publish(self, values: dict[str, float], *, series: str, timestamp: datetime) -> None:
        for sink in self.sinks:
            sink.publish(values, series=series, timestamp=timestamp)
