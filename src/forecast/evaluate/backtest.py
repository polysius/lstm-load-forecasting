"""Rolling-origin backtest: one harness for every model, identical folds/scaling/metrics.

``run_backtest`` produces raw forecasts (``train``); ``score_forecasts`` turns them
into metrics (``evaluate``). Keeping the two apart means scoring can be re-run,
audited or extended without refitting anything.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from forecast.evaluate.metrics import mae, mape, mase_from_scale, mase_scale, rmse
from forecast.evaluate.splits import Fold, generate_folds
from forecast.features.windows import input_windows
from forecast.models.base import Forecaster, predict_many

log = logging.getLogger(__name__)


@dataclass
class FoldInfo:
    fold: Fold
    mase_scale: float
    scaler_mean: float
    scaler_scale: float
    fit_seconds: float
    predict_seconds: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fold"] = self.fold.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FoldInfo:
        return cls(**{**d, "fold": Fold(**d["fold"])})


@dataclass
class BacktestResult:
    model: str
    forecasts: pd.DataFrame
    folds: list[FoldInfo] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.forecasts.to_parquet(path)
        meta = {"model": self.model, "folds": [f.to_dict() for f in self.folds]}
        path.with_suffix(".folds.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> BacktestResult:
        meta = json.loads(path.with_suffix(".folds.json").read_text())
        return cls(
            model=meta["model"],
            forecasts=pd.read_parquet(path),
            folds=[FoldInfo.from_dict(f) for f in meta["folds"]],
        )

    @property
    def fit_seconds(self) -> float:
        return sum(f.fit_seconds for f in self.folds)


def run_backtest(
    model_factory: Callable[[], Forecaster],
    series: pd.Series,
    *,
    input_window: int,
    horizons: list[int] | tuple[int, ...],
    seasonal_period: int,
    n_folds: int = 5,
    covariates: pd.DataFrame | None = None,
    folds: list[Fold] | None = None,
) -> BacktestResult:
    """Fit on each fold's training span; forecast non-overlapping test windows per horizon.

    Per fold: a StandardScaler is fit on the training span only (covariates get
    their own train-only scaler); test windows are cut from the test span with
    stride ``h`` so no target is scored twice for a horizon.
    """
    horizons = list(horizons)
    values = series.to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(series.index)
    cov_values = None if covariates is None else covariates.to_numpy(dtype=np.float64)
    if folds is None:
        folds = generate_folds(
            n_samples=len(values),
            n_folds=n_folds,
            input_window=input_window,
            max_horizon=max(horizons),
        )

    frames: list[pd.DataFrame] = []
    infos: list[FoldInfo] = []
    name = None
    for fold in folds:
        train_vals = values[fold.train_start : fold.train_end]
        test_vals = values[fold.test_start : fold.test_end]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).ravel()
        test_scaled = scaler.transform(test_vals.reshape(-1, 1)).ravel()

        cov_train = cov_test = None
        if cov_values is not None:
            cov_scaler = StandardScaler()
            cov_train = cov_scaler.fit_transform(cov_values[fold.train_start : fold.train_end])
            cov_test = cov_scaler.transform(cov_values[fold.test_start : fold.test_end])

        model = model_factory()
        name = model.name
        t0 = time.perf_counter()
        model.fit(
            train_scaled,
            timestamps=index[fold.train_start : fold.train_end],
            covariates=cov_train,
            horizons=horizons,
        )
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for h in horizons:
            n_windows = len(test_scaled) - input_window - h + 1
            if n_windows <= 0:
                log.warning("fold %d, h=%d: not enough test data for a window", fold.fold_id, h)
                continue
            starts = np.arange(0, n_windows, h)
            X = input_windows(test_scaled, starts, input_window)
            Y = np.stack([test_scaled[s + input_window : s + input_window + h] for s in starts])
            origin_pos = fold.test_start + starts + input_window - 1
            cov_windows = None
            if cov_test is not None:
                cov_windows = np.stack([cov_test[s : s + input_window] for s in starts])
            P = predict_many(model, X, h, origin_times=index[origin_pos], covariates=cov_windows)
            n = len(starts)
            frames.append(
                pd.DataFrame(
                    {
                        "fold": fold.fold_id,
                        "horizon": h,
                        "window": np.repeat(np.arange(n), h),
                        "origin": np.repeat(index[origin_pos], h),
                        "step": np.tile(np.arange(1, h + 1), n),
                        "y_true": Y.ravel(),
                        "y_pred": P.ravel(),
                    }
                )
            )
        infos.append(
            FoldInfo(
                fold=fold,
                mase_scale=mase_scale(train_scaled, seasonal_period),
                scaler_mean=float(scaler.mean_[0]),
                scaler_scale=float(scaler.scale_[0]),
                fit_seconds=fit_s,
                predict_seconds=time.perf_counter() - t0,
            )
        )
        log.info("%s fold %d: fit %.1fs", name, fold.fold_id, fit_s)

    forecasts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return BacktestResult(model=name or "unknown", forecasts=forecasts, folds=infos)


def score_forecasts(result: BacktestResult) -> pd.DataFrame:
    """Per (fold, horizon) metrics on pooled window errors, in the original CSV layout."""
    scale_by_fold = {f.fold.fold_id: f.mase_scale for f in result.folds}
    rows = []
    for (fold_id, h), g in result.forecasts.groupby(["fold", "horizon"], sort=False):
        y_true, y_pred = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        rows.append(
            {
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mape": mape(y_true, y_pred),
                "mase": mase_from_scale(y_true, y_pred, scale_by_fold[fold_id]),
                "model": result.model,
                "horizon": int(h),
                "fold": int(fold_id),
            }
        )
    return pd.DataFrame(rows)
