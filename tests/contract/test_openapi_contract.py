"""API contract: the OpenAPI document is pinned, and live responses must validate against it.

An intentional API change means regenerating the snapshot (``make contract-snapshot``)
and committing it, so every contract change is visible in review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
jsonschema = pytest.importorskip("jsonschema")

from fastapi.testclient import TestClient  # noqa: E402

from forecast.serve.app import create_app  # noqa: E402
from forecast.serve.contract import normalised_openapi  # noqa: E402

SNAPSHOT = Path(__file__).with_name("openapi.snapshot.json")


@pytest.fixture(scope="module")
def client(smoke_bundle):
    with TestClient(create_app(str(smoke_bundle))) as c:
        yield c


def validator(openapi: dict, component: str):
    schema = {"$ref": f"#/components/schemas/{component}", "components": openapi["components"]}
    return jsonschema.Draft202012Validator(schema)


def test_openapi_matches_snapshot():
    current = normalised_openapi(create_app())
    pinned = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert current == pinned, "API contract changed; run `make contract-snapshot` if intended"


def test_contract_shape():
    spec = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert set(spec["paths"]) == {"/health", "/forecast"}
    req = spec["components"]["schemas"]["ForecastRequest"]
    assert set(req["required"]) == {"history", "last_timestamp"}
    resp = spec["components"]["schemas"]["ForecastResponse"]
    assert {"request_id", "model_version", "forecasts", "level"} <= set(resp["required"])


def test_live_responses_validate(client):
    spec = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    health = client.get("/health").json()
    validator(spec, "HealthResponse").validate(health)
    example = spec["components"]["schemas"]["ForecastRequest"]["examples"][0]
    validator(spec, "ForecastRequest").validate(example)
    r = client.post("/forecast", json=example)
    assert r.status_code == 200, r.text
    validator(spec, "ForecastResponse").validate(r.json())
