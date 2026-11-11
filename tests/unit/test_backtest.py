"""The backtest harness, windows, reproduction comparator and configs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.config import load_benchmark_config, load_data_config
from forecast.evaluate.backtest import BacktestResult, run_backtest, score_forecasts
from forecast.evaluate.report import compare_to_reference
from forecast.features.windows import make_windows
from forecast.models.naive import NaiveForecaster, SeasonalNaiveForecaster


def periodic(n: int = 5000, slope: float = 0.01) -> pd.Series:
    """Exact daily cycle plus a linear trend."""
    t = np.arange(n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(50 + 10 * np.sin(2 * np.pi * t / 24) + slope * t, index=idx, name="load")


def _original_windows(y, input_window, max_horizon):
    """The loop from the original models/lstm.py, kept as an oracle."""
    n = len(y) - input_window - max_horizon + 1
    X = np.empty((n, input_window, 1), dtype=np.float32)
    Y = np.empty((n, max_horizon), dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = y[i : i + input_window]
        Y[i, :] = y[i + input_window : i + input_window + max_horizon]
    return X, Y


class TestWindows:
    def test_matches_original_loop(self):
        y = np.random.default_rng(0).normal(size=500)
        X, Y = make_windows(y, 168, 24)
        X0, Y0 = _original_windows(y, 168, 24)
        np.testing.assert_array_equal(X, X0)
        np.testing.assert_array_equal(Y, Y0)

    def test_covariates_cover_the_input_span_only(self):
        y = np.arange(300, dtype=float)
        cov = np.column_stack([np.arange(300) * 10.0])
        X, Y = make_windows(y, 168, 24, covariates=cov)
        assert X.shape == (300 - 192 + 1, 168, 2)
        np.testing.assert_array_equal(X[5, :, 1], np.arange(5, 173) * 10.0)
        assert Y[5, 0] == 173

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="not enough data"):
            make_windows(np.zeros(100), 168, 24)


class TestHarness:
    def test_seasonal_naive_scores_exactly_one_on_cycle_plus_trend(self):
        """The cycle cancels and every error equals the trend's 24h rise - in sample
        (the MASE denominator) and out of sample alike - so MASE must be exactly 1."""
        result = run_backtest(
            SeasonalNaiveForecaster,
            periodic(),
            input_window=168,
            horizons=[1, 6, 24],
            seasonal_period=24,
            n_folds=3,
        )
        scores = score_forecasts(result)
        np.testing.assert_allclose(scores["mase"], 1.0, rtol=1e-9)
        assert list(scores.columns) == ["mae", "rmse", "mape", "mase", "model", "horizon", "fold"]

    def test_windows_are_non_overlapping_per_horizon(self):
        result = run_backtest(
            NaiveForecaster,
            periodic(),
            input_window=168,
            horizons=[6],
            seasonal_period=24,
            n_folds=2,
        )
        f = result.forecasts
        origins = f[f["step"] == 1].groupby("fold")["origin"].apply(lambda s: s.diff().dropna())
        assert (origins == pd.Timedelta(hours=6)).all()

    def test_save_load_roundtrip(self, tmp_path):
        result = run_backtest(
            NaiveForecaster,
            periodic(),
            input_window=168,
            horizons=[1],
            seasonal_period=24,
            n_folds=2,
        )
        result.save(tmp_path / "naive.parquet")
        loaded = BacktestResult.load(tmp_path / "naive.parquet")
        pd.testing.assert_frame_equal(score_forecasts(loaded), score_forecasts(result))
        assert loaded.folds[1].fold == result.folds[1].fold


class TestReproductionComparator:
    def test_pass_and_fail(self):
        ref = pd.DataFrame(
            {
                "model": ["a", "a", "b", "b"],
                "horizon": [1, 1, 1, 1],
                "fold": [0, 1, 0, 1],
                "mase": [1.0, 1.0, 2.0, 2.0],
            }
        )
        rep = ref.copy()
        rep.loc[rep.model == "b", "mase"] = 2.2  # 10% off
        cmp = compare_to_reference(rep, ref, {"a": 0.0, "b": 0.05}).set_index("model")
        assert cmp.loc["a", "status"] == "PASS"
        assert cmp.loc["b", "status"] == "FAIL"
        assert cmp.loc["b", "max_fold_diff"] == pytest.approx(0.2)

    def test_known_differences_are_labelled_not_passed(self):
        ref = pd.DataFrame(
            {"model": ["a", "b"], "horizon": [1, 1], "fold": [0, 0], "mase": [1.0, 2.0]}
        )
        rep = ref.assign(mase=[1.0, 3.0])
        cmp = compare_to_reference(rep, ref, {}, {"a": "doc", "b": "provenance unknown"})
        cmp = cmp.set_index("model")
        assert cmp.loc["a", "status"] == "PASS"  # a waiver never hides a real pass...
        assert cmp.loc["b", "status"] == "KNOWN"  # ...nor turns a miss into a pass
        assert cmp.loc["b", "note"] == "provenance unknown"

    def test_models_absent_from_the_reference_are_skipped(self):
        ref = pd.DataFrame({"model": ["a"], "horizon": [1], "fold": [0], "mase": [1.0]})
        rep = pd.DataFrame(
            {"model": ["a", "z"], "horizon": [1, 1], "fold": [0, 0], "mase": [1.0, 9.0]}
        )
        assert list(compare_to_reference(rep, ref, {})["model"]) == ["a"]


class TestConfigs:
    def test_shipped_configs_load(self):
        assert load_data_config("uci").uci.target_client == "MT_001"
        for name in ("reference", "aligned"):
            cfg = load_benchmark_config(name)
            assert cfg.runs and all(r.models for r in cfg.runs)

    def test_unknown_keys_are_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "name: x\nsource: uci\nuci: {source_url: u, raw_filename: f, "
            "target_client: c}\ntask: {input_windw: 10}\n"
        )
        with pytest.raises(ValueError, match="input_windw"):
            load_data_config(bad)
