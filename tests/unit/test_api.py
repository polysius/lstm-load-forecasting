"""The FastAPI service, in-process, on a real exported bundle."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from forecast.serve.app import create_app  # noqa: E402
from forecast.serve.prediction_log import JsonlLogger, read_jsonl  # noqa: E402

LAST = "2025-04-01T12:00:00Z"


def payload(**overrides):
    t = np.arange(168)
    body = {
        "series_id": "TEST",
        "history": (1000 + 200 * np.sin(2 * np.pi * t / 24)).round(2).tolist(),
        "last_timestamp": LAST,
        "horizon": 24,
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def logdir(tmp_path_factory):
    return tmp_path_factory.mktemp("predictions")


@pytest.fixture(scope="module")
def client(smoke_bundle, logdir):
    app = create_app(str(smoke_bundle), prediction_logger=JsonlLogger(logdir))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["input_window"] == 168 and body["max_horizon"] == 24
    assert body["levels"] == ["0.8", "0.95"]


def test_forecast_happy_path(client):
    r = client.post("/forecast", json=payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["forecasts"]) == 24 and body["level"] == "0.8"
    first = body["forecasts"][0]
    assert first["step"] == 1 and first["timestamp"].startswith("2025-04-01T13:00:00")
    assert all(f["lower"] <= f["point"] <= f["upper"] for f in body["forecasts"])


def test_prediction_is_logged_with_inputs(client, logdir):
    r = client.post("/forecast", json=payload(series_id="LOGME", horizon=6))
    assert r.status_code == 200
    records = [rec for rec in read_jsonl(logdir) if rec["series_id"] == "LOGME"]
    assert len(records) == 1
    rec = records[0]
    assert rec["request_id"] == r.json()["request_id"]
    assert len(rec["inputs"]) == 168 and len(rec["forecasts"]) == 6
    assert rec["origin"].startswith("2025-04-01T12:00:00")


@pytest.mark.parametrize(
    "overrides",
    [
        {"history": [1.0] * 100},
        {"history": [-1.0] * 168},
        {"last_timestamp": "2025-04-01T12:00:00"},
        {"horizon": 48},
        {"horizon": 0},
        {"level": "0.5"},
        {"unexpected": 1},
    ],
    ids=[
        "too-short",
        "negative",
        "naive-ts",
        "horizon-over-model",
        "horizon-zero",
        "uncalibrated-level",
        "extra-field",
    ],
)
def test_invalid_requests_are_422(client, overrides):
    r = client.post("/forecast", json=payload(**overrides))
    assert r.status_code == 422, r.text


def test_nan_is_rejected(client):
    raw = '{"history": [' + ",".join(["1.0"] * 167 + ["NaN"]) + f'], "last_timestamp": "{LAST}"}}'
    r = client.post("/forecast", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    client.get("/health")
    text = client.get("/metrics").text
    assert 'forecast_requests_total{endpoint="/health",status="200"}' in text
    assert "forecast_inference_latency_seconds_bucket" in text
    assert "forecast_model_info" in text


def test_logging_failure_does_not_fail_the_request(smoke_bundle):
    class Broken:
        def log(self, record):
            raise OSError("disk full")

    with TestClient(create_app(str(smoke_bundle), prediction_logger=Broken())) as c:
        assert c.post("/forecast", json=payload()).status_code == 200
        assert "forecast_prediction_log_errors_total 1.0" in c.get("/metrics").text
