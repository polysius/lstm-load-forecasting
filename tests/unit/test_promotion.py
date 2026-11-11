"""The promotion rule: pure decision logic, then the full registry flow (spec M2.4)."""

from __future__ import annotations

import math

import pytest

from forecast.config import PromotionRule
from forecast.train.promote import decide

RULE = PromotionRule(metric="mase_mean", min_improvement=0.0, must_beat_baseline="seasonal_naive")


class TestDecide:
    def test_first_model_must_beat_baseline(self):
        assert decide(0.8, None, 0.9, RULE)[0] is True
        ok, reasons = decide(0.95, None, 0.9, RULE)
        assert ok is False and "does not beat seasonal_naive" in reasons[0]

    def test_must_beat_production(self):
        assert decide(0.70, 0.75, 0.9, RULE)[0] is True
        ok, reasons = decide(0.80, 0.75, 0.9, RULE)
        assert ok is False and any("does not beat production" in r for r in reasons)

    def test_ties_do_not_promote(self):
        assert decide(0.75, 0.75, 0.9, RULE)[0] is False

    def test_min_improvement_margin(self):
        rule = RULE.model_copy(update={"min_improvement": 0.02})
        assert decide(0.74, 0.75, 0.9, rule)[0] is False
        assert decide(0.72, 0.75, 0.9, rule)[0] is True

    def test_non_finite_candidate_rejected(self):
        assert decide(math.nan, 0.75, 0.9, RULE)[0] is False

    def test_baseline_check_can_be_disabled(self):
        rule = RULE.model_copy(update={"must_beat_baseline": None})
        assert decide(1.5, None, None, rule)[0] is True


@pytest.mark.slow
def test_registry_flow_rejects_a_deliberately_worse_model(mlflow_env, tmp_path):
    """Bad v1 -> promoted (no champion); good v2 -> promoted; bad v3 -> rejected."""
    import mlflow

    from forecast.config import ProductionConfig
    from forecast.serve.bundle import read_card
    from forecast.train.production import train_production
    from forecast.train.promote import export_bundle, promote
    from forecast.train.smoke import synthetic_load
    from forecast.train.tracking import Tracker

    series = synthetic_load(n=2400, seed=3)
    tracker = Tracker.from_env()
    base = dict(
        model_name="unit-test-model",
        data="eia",
        experiment="unit-test",
        holdout_hours=168,
        calibration_hours=168,
        seed=0,
    )
    bad = ProductionConfig(
        **base,
        lstm={
            "hidden_size": 2,
            "max_epochs": 1,
            "patience": 1,
            "learning_rate": 1e-5,
            "batch_size": 512,
        },
    )
    good = ProductionConfig(
        **base,
        lstm={
            "hidden_size": 16,
            "max_epochs": 25,
            "patience": 5,
            "learning_rate": 5e-3,
            "batch_size": 128,
        },
    )
    first_rule = PromotionRule(must_beat_baseline=None)

    v1 = train_production(bad, series=series, tracker=tracker, bundle_dir=tmp_path / "b1")
    d1 = promote("unit-test-model", rule=first_rule, version=v1.model_version, series=series)
    assert d1.promote and d1.applied and d1.production_version is None

    v2 = train_production(good, series=series, tracker=tracker, bundle_dir=tmp_path / "b2")
    d2 = promote("unit-test-model", rule=first_rule, version=v2.model_version, series=series)
    assert d2.promote, d2.reasons
    assert d2.candidate < d2.production

    v3 = train_production(bad, series=series, tracker=tracker, bundle_dir=tmp_path / "b3")
    d3 = promote("unit-test-model", rule=first_rule, version=v3.model_version, series=series)
    assert not d3.promote and not d3.applied
    assert any("does not beat production" in r for r in d3.reasons)

    client = mlflow.MlflowClient(tracking_uri=mlflow_env)
    assert str(client.get_model_version_by_alias("unit-test-model", "production").version) == "2"
    assert str(client.get_model_version_by_alias("unit-test-model", "previous").version) == "1"
    tags = client.get_model_version("unit-test-model", "3").tags
    assert tags["promotion.decision"] == "rejected" and tags["promotion.compared_to"] == "2"
    assert len(mlflow.search_runs(search_all_experiments=True)) == 3

    out = export_bundle("unit-test-model", tmp_path / "served", tracker=tracker)
    assert read_card(out).model_version == "2"
