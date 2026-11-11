"""``forecast retrain``: act on a drift trigger only, and obey the promotion gate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from forecast.config import MonitorConfig
from forecast.monitor import retrain
from forecast.monitor.job import MonitorResult, decide_retrain, read_history


def _result(*, trigger: bool, as_of: str = "2024-01-10T00:00:00+00:00") -> MonitorResult:
    return MonitorResult(
        as_of=as_of,
        model_version="1",
        reference={},
        drift={"psi": 1.0, "ks_statistic": 0.5},
        performance={},
        breach=True,
        retrain=trigger,
    )


class _Calls:
    def __init__(self, *, trigger: bool, promote: bool):
        self.trained = 0
        self.exported = 0
        self.trigger, self.promote = trigger, promote

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _result(trigger=self.trigger)
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


def _run(force: bool = False, history: Path = Path("h")):
    return retrain.retrain_if_drifted(
        MonitorConfig(),
        predictions=Path("p"),
        history_path=history,
        model_dir=Path("m"),
        force=force,
    )


def test_no_trigger_does_nothing(monkeypatch, tmp_path):
    calls = _Calls(trigger=False, promote=True)
    calls.install(monkeypatch)
    outcome = _run(history=tmp_path / "h.jsonl")
    assert not outcome.retrained and calls.trained == 0


def test_trigger_retrains_and_refreshes_the_bundle_when_promoted(monkeypatch, tmp_path):
    calls = _Calls(trigger=True, promote=True)
    calls.install(monkeypatch)
    history = tmp_path / "h.jsonl"
    outcome = _run(history=history)
    assert outcome.retrained and outcome.promoted and calls.exported == 1
    assert outcome.model_version == "7"
    assert [h["retrain"] for h in read_history(history)] == [True]


def test_rejected_candidate_leaves_the_served_bundle_alone(monkeypatch, tmp_path):
    calls = _Calls(trigger=True, promote=False)
    calls.install(monkeypatch)
    outcome = _run(history=tmp_path / "h.jsonl")
    assert outcome.retrained and outcome.promoted is False and calls.exported == 0


def test_force_retrains_without_a_trigger(monkeypatch, tmp_path):
    calls = _Calls(trigger=False, promote=False)
    calls.install(monkeypatch)
    assert _run(force=True, history=tmp_path / "h.jsonl").retrained and calls.trained == 1


def test_monitor_check_does_not_start_the_retrain_cooldown(tmp_path):
    history = tmp_path / "h.jsonl"
    day_before = _result(trigger=False, as_of="2024-01-09T00:00:00+00:00")
    retrain.record_check(history, day_before, retrained=False)
    retrain.record_check(history, _result(trigger=True), retrained=False)

    next_day = pd.Timestamp("2024-01-11T00:00:00+00:00")
    should_retrain, _ = decide_retrain(True, read_history(history), MonitorConfig(), next_day, "1")
    assert should_retrain


def test_failed_training_leaves_no_retrain_entry(monkeypatch, tmp_path):
    calls = _Calls(trigger=True, promote=True)
    calls.install(monkeypatch)

    def boom(cfg, tracker=None):
        raise RuntimeError("training failed")

    monkeypatch.setattr("forecast.train.production.train_production", boom)
    history = tmp_path / "h.jsonl"
    with pytest.raises(RuntimeError):
        _run(history=history)
    assert not any(h["retrain"] for h in read_history(history))
