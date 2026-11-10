"""Train a benchmark: every (run, model, seed) through the rolling-origin backtest.

Forecasts are written to ``experiments/forecasts/<benchmark>/<run>/``; scoring
happens in ``forecast.evaluate.report`` so the two steps are independently re-runnable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from forecast.config import (
    BenchmarkConfig,
    ModelSpec,
    RunSpec,
    data_dir,
    experiments_dir,
    load_data_config,
)
from forecast.data.datasets import load_series
from forecast.evaluate.backtest import BacktestResult, run_backtest
from forecast.models import build_model, resolve_params

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    run: RunSpec
    spec: ModelSpec
    seed: int | None

    @property
    def key(self) -> str:
        return self.spec.result_label + ("" if self.seed is None else f"__seed{self.seed}")


def jobs(
    cfg: BenchmarkConfig, runs: list[str] | None = None, models: list[str] | None = None
) -> list[Job]:
    out = []
    for run in cfg.runs:
        if runs and run.name not in runs:
            continue
        for spec in run.models:
            if models and spec.model not in models and spec.result_label not in models:
                continue
            for seed in spec.seeds or (None,):
                out.append(Job(run=run, spec=spec, seed=seed))
    return out


def forecasts_path(benchmark: str, job: Job, root: Path | None = None) -> Path:
    root = root or experiments_dir() / "forecasts"
    return root / benchmark / job.run.name / f"{job.key}.parquet"


def train_benchmark(
    cfg: BenchmarkConfig,
    *,
    runs: list[str] | None = None,
    models: list[str] | None = None,
    skip_existing: bool = False,
    tracker=None,
) -> list[tuple[Job, BacktestResult]]:
    data_cfg = load_data_config(cfg.data)
    series, snap = load_series(data_cfg, data_dir())
    log.info("benchmark %s on %s (%s, %d rows)", cfg.name, snap.dataset, snap.version, len(series))

    done = []
    for job in jobs(cfg, runs, models):
        path = forecasts_path(cfg.name, job)
        if skip_existing and path.exists():
            log.info("skip %s/%s (exists)", job.run.name, job.key)
            continue
        log.info("=== %s / %s (%d folds)", job.run.name, job.key, job.run.n_folds)

        def factory(job: Job = job):
            return build_model(
                job.spec.model,
                job.spec.params,
                task=data_cfg.task,
                seasonal=data_cfg.seasonal,
                seed=job.seed,
            )

        t0 = time.perf_counter()
        result = run_backtest(
            factory,
            series,
            input_window=data_cfg.task.input_window,
            horizons=data_cfg.task.forecast_horizons,
            seasonal_period=data_cfg.seasonal.primary_period,
            n_folds=job.run.n_folds,
        )
        result.model = job.spec.result_label
        result.save(path)
        elapsed = time.perf_counter() - t0
        log.info("saved %s (%.0fs)", path, elapsed)
        if tracker is not None:
            tracker.log_backtest(
                benchmark=cfg.name,
                job=job,
                result=result,
                snapshot=snap,
                data_cfg=data_cfg,
                params=resolve_params(job.spec.model, job.spec.params),
                elapsed=elapsed,
            )
        done.append((job, result))
    return done
