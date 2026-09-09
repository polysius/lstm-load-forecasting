"""Gradient boosting baseline on lag + calendar features (SPEC §6.4)."""

from __future__ import annotations

import warnings

import numpy as np
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


class GBMForecaster:
    def __init__(
        self,
        lag_features: list[int] | None = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 30,
    ):
        cfg_path = _ROOT / "config" / "baselines.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)["gbm"]

        self.lag_features = lag_features or cfg["lag_features"]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self._models: dict[int, object] = {}
        self._train_data: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "gbm"

    def _build_features(self, y: np.ndarray, step_ahead: int) -> tuple[np.ndarray, np.ndarray]:
        """Build lag features for a single step-ahead target.

        Lags must be >= step_ahead to prevent leakage (SPEC §6.4).
        """
        valid_lags = [lag for lag in self.lag_features if lag >= step_ahead]
        if not valid_lags:
            raise ValueError(
                f"No valid lags for step_ahead={step_ahead}. "
                f"All lags must be >= step_ahead. Got lags={self.lag_features}"
            )

        max_lag = max(valid_lags)
        n_samples = len(y) - max_lag - step_ahead + 1
        if n_samples <= 0:
            raise ValueError(f"Not enough data: {len(y)} samples, need > {max_lag + step_ahead}")

        X = np.empty((n_samples, len(valid_lags) + 2))
        targets = np.empty(n_samples)

        for i in range(n_samples):
            t = max_lag + i
            for j, lag in enumerate(valid_lags):
                X[i, j] = y[t - lag]
            X[i, -2] = t % 24
            X[i, -1] = (t // 24) % 7
            targets[i] = y[t + step_ahead - 1]

        return X, targets

    def _build_predict_features(self, y_input: np.ndarray, step_ahead: int) -> np.ndarray:
        valid_lags = [lag for lag in self.lag_features if lag >= step_ahead]
        features = []
        for lag in valid_lags:
            if lag <= len(y_input):
                features.append(y_input[-lag])
            else:
                features.append(y_input[0])
        features.append(len(y_input) % 24)
        features.append((len(y_input) // 24) % 7)
        return np.array(features).reshape(1, -1)

    def fit(self, y_train: np.ndarray, **kwargs) -> None:
        import lightgbm as lgb

        self._train_data = y_train
        horizons = kwargs.get("horizons", [1, 6, 24])
        max_h = max(horizons)

        for step in range(1, max_h + 1):
            try:
                X, y = self._build_features(y_train, step_ahead=step)
            except ValueError:
                continue

            split_idx = int(len(X) * 0.8)
            X_tr, X_val = X[:split_idx], X[split_idx:]
            y_tr, y_val = y[:split_idx], y[split_idx:]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    verbose=-1,
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
                )
            self._models[step] = model

    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray:
        preds = np.empty(horizon)
        for step in range(1, horizon + 1):
            model = self._models.get(step)
            if model is None:
                preds[step - 1] = y_input[-1]
                continue
            X = self._build_predict_features(y_input, step)
            preds[step - 1] = model.predict(X)[0]
        return preds
