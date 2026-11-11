"""Shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def smoke_bundle(tmp_path_factory):
    """A real (tiny) exported bundle: ONNX graph + calibrated model card."""
    pytest.importorskip("torch")
    from forecast.train.smoke import build_smoke_bundle

    return build_smoke_bundle(tmp_path_factory.mktemp("bundle") / "model", epochs=2)


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    """An isolated MLflow tracking store + artefact root for one test."""
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return uri
