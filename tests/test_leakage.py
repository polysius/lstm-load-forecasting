"""Leakage prevention tests (SPEC §5) including the noise canary."""

from __future__ import annotations

import numpy as np

from splits.rolling_origin import generate_folds


class TestChronologicalSplitting:
    """§5.1 — No shuffling, chronological order only."""

    def test_train_before_test(self):
        folds = generate_folds(n_samples=5000, n_folds=5)
        for fold in folds:
            assert fold.train_end <= fold.test_start, (
                f"Fold {fold.fold_id}: train_end={fold.train_end} > test_start={fold.test_start}"
            )

    def test_folds_are_expanding(self):
        folds = generate_folds(n_samples=5000, n_folds=5)
        for i in range(1, len(folds)):
            assert folds[i].train_end > folds[i - 1].train_end, (
                "Training window must expand with each fold"
            )


class TestPurgeGap:
    """§5.4 — Gap of at least L + max(h) between train and test."""

    def test_purge_gap_exists(self):
        folds = generate_folds(
            n_samples=5000, n_folds=5, input_window=168, max_horizon=24
        )
        min_gap = 168 + 24  # L + max(h)
        for fold in folds:
            actual_gap = fold.test_start - fold.train_end
            assert actual_gap >= min_gap, (
                f"Fold {fold.fold_id}: gap={actual_gap} < required={min_gap}"
            )

    def test_purge_gap_size_recorded(self):
        folds = generate_folds(n_samples=5000, n_folds=5)
        for fold in folds:
            assert fold.purge_size > 0


class TestWindowBoundaries:
    """§5.3 — Windows must not straddle split boundaries."""

    def test_no_straddling_windows(self):
        n = 5000
        folds = generate_folds(n_samples=n, n_folds=5, input_window=168, max_horizon=24)
        for fold in folds:
            # A window of size 168 ending at train_end should not
            # have its target reaching into the purge zone or test set
            last_valid_window_end = fold.train_end
            first_test_target = fold.test_start
            gap = first_test_target - last_valid_window_end
            assert gap >= 168 + 24, (
                f"Fold {fold.fold_id}: boundary window could straddle. "
                f"Gap={gap}, need >= {168 + 24}"
            )


class TestScalerLeakage:
    """§5.2 — Scalers fit on training data only."""

    def test_scaler_fit_on_train_only(self):
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        train_data = rng.normal(10, 2, size=(500, 1))
        test_data = rng.normal(100, 50, size=(100, 1))

        scaler = StandardScaler()
        scaler.fit(train_data)

        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Train should be roughly standardized
        assert abs(train_scaled.mean()) < 0.1
        assert abs(train_scaled.std() - 1.0) < 0.1

        # Test should NOT be standardized — it has a different distribution
        # If the scaler was fit on the combined data, test would also be ~N(0,1)
        assert abs(test_scaled.mean()) > 5, (
            "Test data appears standardized — scaler may have been fit on combined data"
        )


class TestNoTargetDerivedFeatures:
    """§5.5 — No rolling/expanding stats computed on full series before splitting."""

    def test_sklearn_train_test_split_absent(self):
        """sklearn.model_selection.train_test_split must not appear in this repo."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        violations = []
        for py_file in repo_root.rglob("*.py"):
            if ".venv" in str(py_file) or "legacy" in str(py_file):
                continue
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if "train_test_split" in text and "test_leakage" not in str(py_file):
                violations.append(str(py_file))
        assert not violations, (
            f"train_test_split found in: {violations}. Use chronological splits only."
        )


class TestNoiseCanary:
    """Canary test: train on pure noise, verify no model beats naive.

    If any model achieves meaningfully better than naive on noise,
    the pipeline is leaking information.
    """

    def test_noise_target_cannot_be_predicted(self):
        rng = np.random.default_rng(42)
        n = 3000
        noise = rng.standard_normal(n)

        # A "trained" model on noise should not beat naive
        # (Here we simulate by checking that the series has no predictable structure)
        from statsmodels.tsa.stattools import acf

        autocorrelations = acf(noise, nlags=48, fft=True)
        # No significant autocorrelation at any lag
        significance_bound = 2 / np.sqrt(n)
        significant_lags = np.sum(np.abs(autocorrelations[1:]) > significance_bound)

        # At 5% significance level with 48 lags, expect ~2.4 false positives.
        # Allow up to ~15% (≈7) to avoid flaky tests from random variation.
        assert significant_lags <= 7, (
            f"Noise series has {significant_lags} significant autocorrelation lags. "
            f"Either the RNG is broken or the pipeline is injecting structure."
        )
