"""Gradient boosting baseline on lag + calendar features (SPEC §6.4)."""

from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


class GBMForecaster:
    def __init__(
        self,
        lag_features: list[int] | None = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 50,
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

    @property
    def name(self) -> str:
        return "gbm"

    def _build_features(self, y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Build lag features ensuring lags >= horizon (SPEC §6.4 leakage guard)."""
        valid_lags = [lag for lag in self.lag_features if lag >= horizon]
        if not valid_lags:
            raise ValueError(
                f"No valid lags for horizon={horizon}. "
                f"All lags must be >= horizon. Got lags={self.lag_features}"
            )

        max_lag = max(valid_lags)
        n_samples = len(y) - max_lag - horizon + 1
        if n_samples <= 0:
            raise ValueError(f"Not enough data: {len(y)} samples, need > {max_lag + horizon}")

        X = np.empty((n_samples, len(valid_lags) + 2))
        targets = np.empty(n_samples)

        for i in range(n_samples):
            t = max_lag + i
            for j, lag in enumerate(valid_lags):
                X[i, j] = y[t - lag]
            # Calendar-like features from position (hour-of-day proxy)
            X[i, -2] = t % 24
            X[i, -1] = (t // 24) % 7
            targets[i] = y[t + horizon - 1]

        return X, targets

    def fit(self, y_train: np.ndarray, **kwargs) -> None:
        import lightgbm as lgb

        horizons = kwargs.get("horizons", [1, 6, 24])
        for h in horizons:
            try:
                X, y = self._build_features(y_train, horizon=h)
            except ValueError as e:
                print(f"  GBM h={h}: {e}")
                continue

            split_idx = int(len(X) * 0.8)
            X_tr, X_val = X[:split_idx], X[split_idx:]
            y_tr, y_val = y[:split_idx], y[split_idx:]

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
            self._models[h] = model

    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray:
        model = self._models.get(horizon)
        if model is None:
            return np.full(horizon, y_input[-1])

        valid_lags = [lag for lag in self.lag_features if lag >= horizon]
        features = []
        for lag in valid_lags:
            if lag <= len(y_input):
                features.append(y_input[-lag])
            else:
                features.append(y_input[0])
        features.append(len(y_input) % 24)
        features.append((len(y_input) // 24) % 7)

        X = np.array(features).reshape(1, -1)
        pred = model.predict(X)
        return np.full(horizon, pred[0])
