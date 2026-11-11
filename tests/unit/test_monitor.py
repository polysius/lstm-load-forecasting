"""Drift scores, prediction/actual joins, the retrain decision and live-traffic windows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from forecast.config import MonitorConfig
from forecast.monitor.drift import psi
from forecast.monitor.job import decide_retrain, reference_window, run_monitor
from forecast.monitor.performance import join_actuals, predictions_frame, rolling_metrics
from forecast.monitor.traffic import latest_window
from forecast.serve.bundle import IntervalSpec, ModelCard, ScalerParams


def card(**metrics) -> ModelCard:
    return ModelCard(
        model_name="m",
        model_version="3",
        created_at="2026-01-01T00:00:00+00:00",
        architecture={},
        input_window=168,
        max_horizon=24,
        scaler=ScalerParams(mean=0, scale=1),
        intervals=IntervalSpec(levels={"0.8": [1.0] * 24}, default_level="0.8"),
        training={"holdout_end": "2026-03-01T00:00:00+00:00"},
        metrics={"mase_scale": 10.0, **metrics},
    )


def two_years(seed=0, shift_from=None, factor=1.25, annual=100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    t = np.arange(len(idx))
    y = 1000 + 200 * np.sin(2 * np.pi * t / 24) + annual * np.sin(2 * np.pi * t / (24 * 365))
    y = y + rng.normal(0, 20, len(idx))
    s = pd.Series(y, index=idx)
    if shift_from is not None:
        s[s.index >= pd.Timestamp(shift_from, tz="UTC")] *= factor
    return s


class TestPSI:
    def test_same_distribution_is_near_zero(self):
        rng = np.random.default_rng(0)
        assert psi(rng.normal(size=5000), rng.normal(size=2000)) < 0.02

    def test_shift_is_large_and_monotone(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(size=5000)
        small, big = psi(ref, rng.normal(0.3, 1, 2000)), psi(ref, rng.normal(1.5, 1, 2000))
        assert 0.02 < small < big and big > 1.0

    def test_agrees_with_evidently(self):
        pytest.importorskip("evidently")
        from forecast.monitor.drift import drift_scores

        rng = np.random.default_rng(2)
        res = drift_scores(rng.normal(100, 10, 2000), rng.normal(108, 10, 168))
        assert res.psi == pytest.approx(res.psi_evidently, rel=0.35)
        assert res.ks_statistic > 0.2 and res.ks_pvalue < 1e-3

    def test_quantile_bins_see_a_shift_that_equal_width_bins_miss(self):
        """Regression for the planted-shift demo: rare spikes (p99 5, max 30) put
        nearly all mass in Evidently's first equal-width bin."""
        pytest.importorskip("evidently")
        from forecast.monitor.drift import drift_scores

        rng = np.random.default_rng(3)

        def spiky(n, scale=1.0):
            base = rng.lognormal(np.log(1.9), 0.25, n)
            base[rng.random(n) < 0.004] = rng.uniform(15, 30)
            return base * scale

        res = drift_scores(spiky(672), spiky(168, scale=1.25))
        assert res.psi > 0.5, res
        assert res.psi_evidently < 0.2, res  # the failure mode this guards against


class TestPerformanceJoin:
    def _records(self, origin: datetime, error: float):
        return [
            {
                "request_id": "r1",
                "model_version": "3",
                "origin": origin.isoformat(),
                "forecasts": [
                    {
                        "step": k,
                        "timestamp": (origin + timedelta(hours=k)).isoformat(),
                        "point": 100 + error,
                        "lower": 100 + error - 5,
                        "upper": 100 + error + 5,
                    }
                    for k in range(1, 25)
                ],
            }
        ]

    def test_mase_and_coverage(self):
        origin = datetime(2026, 1, 1, tzinfo=UTC)
        idx = pd.date_range(origin, periods=48, freq="h", tz="UTC")
        actuals = pd.Series(100.0, index=idx)
        joined = join_actuals(predictions_frame(self._records(origin, error=4.0)), actuals)
        m = rolling_metrics(
            joined,
            start=pd.Timestamp(origin),
            end=pd.Timestamp(origin) + pd.Timedelta(hours=1),
            scale=2.0,
        )
        assert m["mae_h1"] == pytest.approx(4.0) and m["mase_h24"] == pytest.approx(2.0)
        assert m["coverage"] == 1.0 and m["n_requests"] == 1

    def test_unobserved_targets_are_not_scored(self):
        origin = datetime(2026, 1, 1, tzinfo=UTC)
        actuals = pd.Series(100.0, index=pd.date_range(origin, periods=6, freq="h", tz="UTC"))
        joined = join_actuals(predictions_frame(self._records(origin, error=1.0)), actuals)
        m = rolling_metrics(
            joined,
            start=pd.Timestamp(origin),
            end=pd.Timestamp(origin) + pd.Timedelta(hours=1),
            scale=1.0,
        )
        assert m["n_scored_points"] == 5  # steps 1..5 observed


class TestDecision:
    cfg = MonitorConfig(consecutive_breaches=2, cooldown_hours=72)
    t = pd.Timestamp("2026-02-10", tz="UTC")

    def hist(self, *entries):
        return [
            {"as_of": (self.t - pd.Timedelta(days=d)).isoformat(), "breach": b, "retrain": r}
            for d, b, r in entries
        ]

    def test_first_breach_is_debounced(self):
        assert decide_retrain(True, self.hist((1, False, False)), self.cfg, self.t)[0] is False

    def test_second_consecutive_breach_retrains(self):
        assert decide_retrain(True, self.hist((1, True, False)), self.cfg, self.t)[0] is True

    def test_cooldown_blocks_back_to_back_retrains(self):
        ok, why = decide_retrain(
            True, self.hist((2, True, True), (1, True, False)), self.cfg, self.t
        )
        assert ok is False and "cooldown" in why[0]

    def test_a_new_model_starts_a_fresh_streak(self):
        hist = [
            {
                "as_of": (self.t - pd.Timedelta(days=1)).isoformat(),
                "breach": True,
                "retrain": False,
                "model_version": "1",
            }
        ]
        assert decide_retrain(True, hist, self.cfg, self.t, model_version="1")[0] is True
        ok, why = decide_retrain(True, hist, self.cfg, self.t, model_version="2")
        assert ok is False and "1/2" in why[0]  # v2 is not blamed for v1's breaches

    def test_no_breach_never_retrains(self):
        assert decide_retrain(False, self.hist((1, True, False)), self.cfg, self.t)[0] is False


class TestRunMonitor:
    def test_only_the_deployed_models_predictions_are_scored(self):
        as_of = pd.Timestamp("2025-06-30", tz="UTC")
        series = two_years()
        origin = (as_of - pd.Timedelta(days=3)).to_pydatetime()
        rec = TestPerformanceJoin()._records(origin, error=50.0)[0]
        old_model, current = dict(rec, model_version="1"), dict(rec, model_version="3")
        cfg = MonitorConfig(psi_threshold=0.2)
        only_old = run_monitor(cfg, actuals=series, records=[old_model], card=card(), as_of=as_of)
        assert only_old.performance["n_requests"] == 0  # card() is model version 3
        both = run_monitor(
            cfg, actuals=series, records=[old_model, current], card=card(), as_of=as_of
        )
        assert both.performance["n_requests"] == 1

    def test_reference_is_the_same_weeks_last_year(self):
        start, end = reference_window(MonitorConfig(), pd.Timestamp("2025-06-30", tz="UTC"), card())
        assert (end - start) == pd.Timedelta(days=28)
        assert (
            pd.Timestamp("2024-06-27", tz="UTC")
            < (start + (end - start) / 2)
            < pd.Timestamp("2024-07-03", tz="UTC")
        )

    @pytest.mark.parametrize("shifted", [False, True])
    def test_auto_threshold_is_calibrated_on_clean_history(self, shifted):
        # No annual cycle: with one, a level shift can partly cancel a seasonal decline
        # and the test would measure the confound, not the detector.
        as_of = pd.Timestamp("2025-06-30", tz="UTC")
        series = two_years(shift_from="2025-06-25" if shifted else None, annual=0.0)
        cfg = MonitorConfig(psi_threshold="auto", reference="training_tail")
        c = card()
        c.training["holdout_end"] = "2025-06-01T00:00:00+00:00"
        result = run_monitor(cfg, actuals=series, records=[], card=c, as_of=as_of)
        assert result.threshold >= cfg.psi_threshold_floor
        assert result.threshold_source.startswith("auto")
        assert result.breach is shifted

    @pytest.mark.parametrize("shifted", [False, True])
    def test_planted_shift_breaches_and_clean_data_does_not(self, shifted):
        as_of = pd.Timestamp("2025-06-30", tz="UTC")
        series = two_years(shift_from="2025-06-25" if shifted else None)
        cfg = MonitorConfig(psi_threshold=0.2)
        history = [
            {
                "as_of": (as_of - pd.Timedelta(days=1)).isoformat(),
                "breach": shifted,
                "retrain": False,
                "model_version": "3",
            }
        ]
        result = run_monitor(
            cfg, actuals=series, records=[], card=card(), history=history, as_of=as_of
        )
        assert result.breach is shifted
        assert result.retrain is shifted  # second consecutive breach


def test_latest_window_fills_short_gaps_only():
    idx = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    s = pd.Series(np.arange(200, dtype=float), index=idx)
    s.iloc[150:152] = np.nan  # 2-hour gap: filled
    values, last = latest_window(s, 168)
    assert len(values) == 168 and last == idx[-1].to_pydatetime()
    s.iloc[180:186] = np.nan  # 6-hour gap inside the window: refused
    with pytest.raises(ValueError):
        latest_window(s, 168)
