"""``forecast retrain``: act on a drift trigger only, and obey the promotion gate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from forecast.config import MonitorConfig
from forecast.monitor import retrain


class _Calls:
    def __init__(self, *, trigger: bool, promote: bool):
        self.trained = 0
        self.exported = 0
        self.trigger, self.promote = trigger, promote

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = SimpleNamespace(retrain=self.trigger)
        monkeypatch.setattr(retrain, "check_drift", lambda *a, **k: result)
        monkeypatch.setattr(retrain, "load_production_config", lambda name: _production())
        monkeypatch.setattr("forecast.train.tracking.Tracker.from_env", lambda: None)
        monkeypatch.setattr("forecast.train.production.train_production", self._train)
        monkeypatch.setattr("forecast.train.promote.promote", self._promote)
        monkeypatch.setattr("forecast.train.promote.export_bundle", self._export)

    def _train(self, cfg, tracker=None):
        self.trained += 1
        return SimpleNamespace(model_version="7")

    def _promote(self, name, rule):
        return SimpleNamespace(promote=self.promote, reasons=["test"])

    def _export(self, name, out_dir, alias):
        self.exported += 1


def _production():
    return SimpleNamespace(model_name="m", promotion=None)


def _run(force: bool = False):
    return retrain.retrain_if_drifted(
        MonitorConfig(),
        predictions=Path("p"),
        history_path=Path("h"),
        model_dir=Path("m"),
        force=force,
    )


def test_no_trigger_does_nothing(monkeypatch):
    calls = _Calls(trigger=False, promote=True)
    calls.install(monkeypatch)
    outcome = _run()
    assert not outcome.retrained and calls.trained == 0


def test_trigger_retrains_and_refreshes_the_bundle_when_promoted(monkeypatch):
    calls = _Calls(trigger=True, promote=True)
    calls.install(monkeypatch)
    outcome = _run()
    assert outcome.retrained and outcome.promoted and calls.exported == 1
    assert outcome.model_version == "7"


def test_rejected_candidate_leaves_the_served_bundle_alone(monkeypatch):
    calls = _Calls(trigger=True, promote=False)
    calls.install(monkeypatch)
    outcome = _run()
    assert outcome.retrained and outcome.promoted is False and calls.exported == 0


def test_force_retrains_without_a_trigger(monkeypatch):
    calls = _Calls(trigger=False, promote=False)
    calls.install(monkeypatch)
    assert _run(force=True).retrained and calls.trained == 1
