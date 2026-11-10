"""MLflow tracking: every run logs params, metrics, artefacts and the data version hash.

Defaults keep everything local (SQLite backend + a local artefact root under
``mlruns/``); in AWS the retraining job points ``MLFLOW_TRACKING_URI`` at a SQLite
file synced from S3 and ``MLFLOW_ARTIFACT_ROOT`` at ``s3://<bucket>/mlflow``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forecast.config import project_root

if TYPE_CHECKING:
    from forecast.config import DataConfig
    from forecast.data.versioning import Snapshot
    from forecast.evaluate.backtest import BacktestResult
    from forecast.train.benchmark import Job

log = logging.getLogger(__name__)


def default_tracking_uri() -> str:
    path = project_root() / "mlruns" / "mlflow.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def default_artifact_root() -> str:
    return str(project_root() / "mlruns" / "artifacts")


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("GITHUB_SHA", "unknown")[:12]


def data_tags(snapshot: Snapshot) -> dict[str, str]:
    return {
        "data.dataset": snapshot.dataset,
        "data.version": snapshot.version,
        "data.sha256": snapshot.sha256,
        "data.rows": str(snapshot.rows),
        "data.range": f"{snapshot.start} -> {snapshot.end}",
    }


@dataclass
class Tracker:
    tracking_uri: str
    artifact_root: str
    experiment_prefix: str = ""

    @classmethod
    def from_env(cls) -> Tracker:
        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", default_tracking_uri()),
            artifact_root=os.environ.get("MLFLOW_ARTIFACT_ROOT", default_artifact_root()),
            experiment_prefix=os.environ.get("MLFLOW_EXPERIMENT_PREFIX", ""),
        )

    def _mlflow(self):
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        return mlflow

    def experiment(self, name: str) -> str:
        """Create (with a controlled artefact location) or fetch an experiment; return its id."""
        mlflow = self._mlflow()
        full = f"{self.experiment_prefix}{name}"
        exp = mlflow.get_experiment_by_name(full)
        if exp is None:
            location = f"{self.artifact_root.rstrip('/')}/{full}"
            exp_id = mlflow.create_experiment(full, artifact_location=location)
        else:
            exp_id = exp.experiment_id
        mlflow.set_experiment(experiment_id=exp_id)
        return exp_id

    @contextmanager
    def run(self, experiment: str, run_name: str, tags: dict[str, str]) -> Iterator[Any]:
        mlflow = self._mlflow()
        self.experiment(experiment)
        with mlflow.start_run(
            run_name=run_name, tags={"git.revision": git_revision(), **tags}
        ) as run:
            yield run

    @staticmethod
    def log_dataset(snapshot: Snapshot, frame, context: str = "training") -> None:
        import mlflow

        dataset = mlflow.data.from_pandas(
            frame.reset_index(),
            source=snapshot.source or snapshot.path,
            name=f"{snapshot.dataset}@{snapshot.version}",
            digest=snapshot.sha256[:16],
        )
        mlflow.log_input(dataset, context=context)

    # ------------------------------------------------------------------ benchmark runs

    def log_backtest(
        self,
        *,
        benchmark: str,
        job: Job,
        result: BacktestResult,
        snapshot: Snapshot,
        data_cfg: DataConfig,
        params: dict[str, Any],
        elapsed: float,
    ) -> str:
        import mlflow

        from forecast.evaluate.backtest import score_forecasts

        scores = score_forecasts(result)
        tags = {
            "kind": "backtest",
            "benchmark": benchmark,
            "benchmark.run": job.run.name,
            "model": job.spec.model,
            "label": job.spec.result_label,
            **data_tags(snapshot),
        }
        if job.seed is not None:
            tags["seed"] = str(job.seed)
        with self.run(f"benchmark-{benchmark}", f"{job.run.name}/{job.key}", tags) as run:
            mlflow.log_params(
                {
                    **{f"model.{k}": v for k, v in params.items()},
                    "backtest.n_folds": job.run.n_folds,
                    "task.input_window": data_cfg.task.input_window,
                    "task.horizons": list(data_cfg.task.forecast_horizons),
                    "seasonal_period": data_cfg.seasonal.primary_period,
                    "seed": job.seed,
                }
            )
            for h, g in scores.groupby("horizon"):
                mlflow.log_metrics(
                    {
                        f"mase_h{h}": float(g["mase"].mean()),
                        f"mae_z_h{h}": float(g["mae"].mean()),
                        f"rmse_z_h{h}": float(g["rmse"].mean()),
                    }
                )
                for row in g.itertuples():  # per-fold curve, fold id as the step
                    mlflow.log_metric(f"fold_mase_h{h}", float(row.mase), step=int(row.fold))
            mlflow.log_metrics(
                {
                    "mase_mean": float(scores["mase"].mean()),
                    "fit_seconds": result.fit_seconds,
                    "elapsed_seconds": elapsed,
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{job.key}.parquet"
                result.save(path)
                mlflow.log_artifact(str(path), artifact_path="forecasts")
                mlflow.log_artifact(str(path.with_suffix(".folds.json")), artifact_path="forecasts")
            return run.info.run_id
