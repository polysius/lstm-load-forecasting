"""Gradient-boosting baseline: one LightGBM model per step ahead on lag + calendar features.

Two feature alignments are supported:

``legacy`` (reproduces the original study, bugs included)
    * Calendar features are positions (``t % 24``, ``(t // 24) % 7``) at training
      time but the constant ``len(window) % 24 == 0`` at predict time - a
      train/serve skew that feeds every forecast "hour 0, day 0".
    * Lag ``L`` for step ``s`` reads ``y[origin + 1 - L]`` instead of
      ``y[target - L]``, so for ``s > 1`` the daily/weekly lags are misaligned by
      ``s - 1`` hours (e.g. step 24 sees ``target - 47`` instead of ``target - 24``).

``target`` (the fix)
    * Lag ``L`` reads ``y[target - L]`` and is used only when ``L >= s``, i.e.
      when that value is observed at the forecast origin.
    * Calendar features are the true hour-of-day and day-of-week of the target
      timestamp, computed identically at fit and predict time.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd


class GBMForecaster:
    name = "gbm"

    def __init__(
        self,
        lag_features: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 168),
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 30,
        val_fraction: float = 0.2,
        feature_alignment: Literal["legacy", "target"] = "legacy",
        n_jobs: int | None = None,
        seed: int | None = None,
    ):
        if feature_alignment not in ("legacy", "target"):
            raise ValueError(f"unknown feature_alignment {feature_alignment!r}")
        self.lag_features = tuple(lag_features)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.feature_alignment = feature_alignment
        self.n_jobs = n_jobs
        self.seed = seed
        self._models: dict[int, Any] = {}

    def _valid_lags(self, step: int) -> list[int]:
        lags = [lag for lag in self.lag_features if lag >= step]
        if not lags:
            raise ValueError(f"no lag >= step {step} in {self.lag_features}")
        return lags

    # ------------------------------------------------------------------ features

    def _legacy_train_features(self, y: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
        lags = self._valid_lags(step)
        max_lag = max(lags)
        n = len(y) - max_lag - step + 1
        if n <= 0:
            raise ValueError(f"not enough data: {len(y)} samples, need > {max_lag + step}")
        t = max_lag + np.arange(n)
        cols = [y[t - lag] for lag in lags] + [t % 24, (t // 24) % 7]
        return np.column_stack(cols).astype(float), y[t + step - 1]

    def _legacy_predict_features(self, windows: np.ndarray, step: int) -> np.ndarray:
        n, length = windows.shape
        cols = [
            windows[:, -lag] if lag <= length else windows[:, 0] for lag in self._valid_lags(step)
        ]
        cols += [np.full(n, length % 24), np.full(n, (length // 24) % 7)]
        return np.column_stack(cols).astype(float)

    @staticmethod
    def _calendar(ts: pd.DatetimeIndex) -> np.ndarray:
        return np.column_stack([ts.hour, ts.day_of_week]).astype(float)

    def _target_train_features(
        self, y: np.ndarray, ts: pd.DatetimeIndex, step: int, cov: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        lags = self._valid_lags(step)
        target_idx = np.arange(max(lags), len(y))
        cols = [y[target_idx - lag] for lag in lags]
        feats = [np.column_stack(cols), self._calendar(ts[target_idx])]
        if cov is not None:  # covariates observed at the forecast origin
            feats.append(cov[target_idx - step])
        return np.hstack(feats), y[target_idx]

    def _target_predict_features(
        self,
        windows: np.ndarray,
        step: int,
        origin_times: pd.DatetimeIndex,
        cov_windows: np.ndarray | None,
    ) -> np.ndarray:
        length = windows.shape[1]
        origin = length - 1
        cols = [windows[:, origin + step - lag] for lag in self._valid_lags(step)]
        target_times = origin_times + pd.Timedelta(hours=step)
        feats = [np.column_stack(cols), self._calendar(target_times)]
        if cov_windows is not None:
            feats.append(cov_windows[:, -1, :])
        return np.hstack(feats)

    # ------------------------------------------------------------------ API

    def fit(self, y_train: np.ndarray, **context: Any) -> GBMForecaster:
        import lightgbm as lgb

        horizons = context.get("horizons") or [1, 6, 24]
        timestamps = context.get("timestamps")
        covariates = context.get("covariates")
        if self.feature_alignment == "target" and timestamps is None:
            raise ValueError("feature_alignment='target' needs the training timestamps")

        self._models = {}
        for step in range(1, max(horizons) + 1):
            try:
                if self.feature_alignment == "legacy":
                    X, y = self._legacy_train_features(y_train, step)
                else:
                    X, y = self._target_train_features(
                        y_train, pd.DatetimeIndex(timestamps), step, covariates
                    )
            except ValueError:
                continue
            split = int(len(X) * (1.0 - self.val_fraction))
            params: dict[str, Any] = dict(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                verbose=-1,
            )
            if self.n_jobs is not None:
                params["n_jobs"] = self.n_jobs
            if self.seed is not None:
                params["random_state"] = self.seed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X[:split],
                    y[:split],
                    eval_set=[(X[split:], y[split:])],
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
                )
            self._models[step] = model
        return self

    def predict_batch(self, windows: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        n = windows.shape[0]
        out = np.empty((n, horizon))
        origin_times = context.get("origin_times")
        cov_windows = context.get("covariates")
        for step in range(1, horizon + 1):
            model = self._models.get(step)
            if model is None:
                out[:, step - 1] = windows[:, -1]
                continue
            if self.feature_alignment == "legacy":
                X = self._legacy_predict_features(windows, step)
            else:
                if origin_times is None:
                    raise ValueError("feature_alignment='target' needs origin_times")
                X = self._target_predict_features(
                    windows, step, pd.DatetimeIndex(origin_times), cov_windows
                )
            out[:, step - 1] = model.predict(X)
        return out

    def predict(self, y_input: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        ctx = dict(context)
        if ctx.get("origin_times") is None and ctx.get("origin_time") is not None:
            ctx["origin_times"] = pd.DatetimeIndex([ctx.pop("origin_time")])
        if ctx.get("covariates") is not None:
            ctx["covariates"] = np.asarray(ctx["covariates"])[None, ...]
        return self.predict_batch(np.asarray(y_input)[None, :], horizon, **ctx)[0]
