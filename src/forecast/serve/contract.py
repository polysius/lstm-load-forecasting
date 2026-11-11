"""Helpers for the pinned OpenAPI contract."""

from __future__ import annotations

import json
from pathlib import Path


def normalised_openapi(app) -> dict:
    """The app's OpenAPI document minus fields that change without a contract change."""
    spec = json.loads(json.dumps(app.openapi()))
    spec.get("info", {}).pop("version", None)
    return spec


def write_snapshot(app, path: Path) -> Path:
    path.write_text(
        json.dumps(normalised_openapi(app), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
