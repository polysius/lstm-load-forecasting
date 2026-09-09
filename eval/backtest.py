"""Rolling-origin backtesting harness (SPEC §8.1).

All models are evaluated through this single harness to ensure identical folds,
scaling, and metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from eval.metrics import compute_all_metrics
from splits.rolling_origin import generate_folds

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "config" / "data.yaml"
_RESULTS_DIR = _ROOT / "experiments" / "results"


def load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


class Forecaster(Protocol):
    """Interface that all models must implement."""

    def fit(self, y_train: np.ndarray, **kwargs) -> None: ...
    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray: ...

    @property
    def name(self) -> str: ...


def run_backtest(
    model: Forecaster,
    df: pd.DataFrame,
    target_col: str = "load_kw",
    n_folds: int = 5,
    horizons: list[int] | None = None,
    seasonal_period: int | None = None,
) -> pd.DataFrame:
    """Run rolling-origin backtest for a single model across all horizons.

    Returns a DataFrame with columns: model, horizon, fold, mae, rmse, mape, mase.
    """
    cfg = load_config()
    if horizons is None:
        horizons = cfg["task"]["forecast_horizons"]
    if seasonal_period is None:
        seasonal_period = cfg["seasonal"]["primary_period"]

    input_window = cfg["task"]["input_window"]
    max_horizon = max(horizons)

    values = df[target_col].values.astype(np.float64)
    folds = generate_folds(
        n_samples=len(values),
        n_folds=n_folds,
        input_window=input_window,
        max_horizon=max_horizon,
    )

    results = []
    for fold in folds:
        train_vals = values[fold.train_start : fold.train_end]
        test_vals = values[fold.test_start : fold.test_end]

        # §5.2 — Scaler fit on training data only
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).ravel()
        test_scaled = scaler.transform(test_vals.reshape(-1, 1)).ravel()

        model.fit(train_scaled)

        for h in horizons:
            y_true_list = []
            y_pred_list = []

            n_windows = len(test_scaled) - input_window - h + 1
            if n_windows <= 0:
                print(
                    f"  Warning: fold {fold.fold_id}, h={h}: not enough test data for any window"
                )
                continue

            for i in range(0, n_windows, h):
                start = i
                end = start + input_window
                if end + h > len(test_scaled):
                    break

                x = test_scaled[start:end]
                y_true = test_scaled[end : end + h]
                y_pred = model.predict(x, horizon=h)

                if len(y_pred) != h:
                    y_pred = y_pred[:h]

                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

            if not y_true_list:
                continue

            y_true_all = np.concatenate(y_true_list)
            y_pred_all = np.concatenate(y_pred_list)

            metrics = compute_all_metrics(
                y_true_all, y_pred_all, train_scaled, seasonal_period
            )
            metrics["model"] = model.name
            metrics["horizon"] = h
            metrics["fold"] = fold.fold_id
            results.append(metrics)

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, filename: str) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved results to {path}")
    return path
