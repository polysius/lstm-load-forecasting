"""The planted-shift demo's configuration (the demo itself runs against Docker)."""

from __future__ import annotations

import pytest

from forecast.config import load_monitor_config, load_production_config
from forecast.monitor.simulate import SimulationConfig, _restart_args, load_simulation_config


@pytest.mark.parametrize("name", ["simulate", "simulate_uci"])
def test_shipped_demo_configs_resolve(name):
    sim = load_simulation_config(name)
    assert 0 < sim.shift_day < sim.days and sim.shift_factor != 1.0
    prod = load_production_config(sim.production)
    assert prod.holdout_hours + prod.calibration_hours < sim.shift_day * 24
    load_monitor_config(sim.monitor)


def test_explicit_path_is_used(tmp_path):
    path = tmp_path / "local.yaml"
    path.write_text("shift_factor: 1.5\nseries_label: X\n")
    sim = load_simulation_config(str(path))
    assert sim.shift_factor == 1.5 and sim.series_label == "X"


def test_missing_explicit_path_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_simulation_config(str(tmp_path / "nope.yaml"))


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("shift_factr: 1.5\n")
    with pytest.raises(ValueError, match="shift_factr"):
        load_simulation_config(str(path))


def test_docker_binary_can_be_overridden(monkeypatch):
    sim = SimulationConfig(restart_command="docker compose restart api")
    monkeypatch.delenv("DOCKER_BIN", raising=False)
    assert _restart_args(sim)[0] == "docker"
    monkeypatch.setenv("DOCKER_BIN", "docker.exe")
    assert _restart_args(sim) == ["docker.exe", "compose", "restart", "api"]
