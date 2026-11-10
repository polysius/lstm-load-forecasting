"""Black-box tests against a running container (CI starts it; see .github/workflows/ci.yml).

FORECAST_API_URL=http://localhost:8080 pytest -m integration
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
jsonschema = pytest.importorskip("jsonschema")

pytestmark = pytest.mark.integration
BASE = os.environ.get("FORECAST_API_URL")
SNAPSHOT = Path(__file__).parents[1] / "contract" / "openapi.snapshot.json"

if not BASE:
    pytest.skip("FORECAST_API_URL not set", allow_module_level=True)


@pytest.fixture(scope="module")
def spec():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=BASE, timeout=30) as c:
        yield c


def validate(spec, component, instance):
    schema = {"$ref": f"#/components/schemas/{component}", "components": spec["components"]}
    jsonschema.Draft202012Validator(schema).validate(instance)


def test_health(http, spec):
    r = http.get("/health")
    assert r.status_code == 200
    validate(spec, "HealthResponse", r.json())


def test_forecast(http, spec):
    example = spec["components"]["schemas"]["ForecastRequest"]["examples"][0]
    r = http.post("/forecast", json=example)
    assert r.status_code == 200, r.text
    body = r.json()
    validate(spec, "ForecastResponse", body)
    assert len(body["forecasts"]) == example["horizon"]


def test_bad_request_is_422(http):
    r = http.post("/forecast", json={"history": [1.0], "last_timestamp": "2026-01-01T00:00:00Z"})
    assert r.status_code == 422


def test_metrics_exposed(http):
    assert "forecast_requests_total" in http.get("/metrics").text
