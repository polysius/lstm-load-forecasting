"""Train the deployable model: fit, calibrate intervals, export ONNX, score, log, register.

Time layout of the latest snapshot (oldest -> newest)::

    [ train ........................ | calibration | holdout ]
                                       (conformal)   (metrics + promotion)

The scaler and LSTM see only ``train``; interval half-widths come from
``calibration``; ``holdout`` is never used for fitting. The artefact that is
evaluated on the holdout is exactly the one that gets registered and served.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.config import ProductionConfig, data_dir, load_data_config
from forecast.data.datasets import load_series
from forecast.evaluate.holdout import (
    baseline_metrics,
    evaluate_predictor,
    rolling_windows,
    seasonal_scale,
)
from forecast.features import calendar_np
from forecast.models import build_model, resolve_params
from forecast.serve.bundle import MODEL_FILE, IntervalSpec, ModelCard, ScalerParams, write_card
from forecast.serve.predictor import Predictor
from forecast.train.conformal import conformal_halfwidths, level_key
from forecast.train.export import export_with_parity

log = logging.getLogger(__name__)


@dataclass
class ProductionResult:
    bundle_dir: Path
    card: ModelCard
    run_id: str | None = None
    model_version: str | None = None
    holdout: dict[str, float] = field(default_factory=dict)
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def train_production(
    cfg: ProductionConfig,
    *,
    data_version: str | None = None,
    until: datetime | pd.Timestamp | None = None,
    series: pd.Series | None = None,
    tracker=None,
    bundle_dir: Path | None = None,
    register: bool = True,
    extra_tags: dict[str, str] | None = None,
) -> ProductionResult:
    data_cfg = load_data_config(cfg.data)
    if series is None:
        series, snap = load_series(data_cfg, data_dir(), version=data_version)
    else:
        snap = None
    if until is not None:
        cutoff = pd.Timestamp(until)
        if cutoff.tz is None:
            cutoff = cutoff.tz_localize(series.index.tz)
        series = series[series.index < cutoff]

    values = series.to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(series.index)
    hours = calendar_np.epoch_hours(index)
    task = data_cfg.task
    L, H = task.input_window, task.max_horizon
    n = len(values)
    hs = n - cfg.holdout_hours
    cs = hs - cfg.calibration_hours
    ts = 0 if cfg.train_window_hours is None else max(0, cs - cfg.train_window_hours)
    if cs - ts < 4 * (L + H):
        raise ValueError(f"only {cs - ts} training hours before calibration; need >= {4 * (L + H)}")

    train = values[ts:cs]
    mean, std = float(np.nanmean(train)), float(np.nanstd(train))
    scaled = (values - mean) / std
    cov = calendar_np.encode(hours, cfg.covariates) if cfg.covariates else None

    params = resolve_params("lstm", cfg.lstm)
    model = build_model("lstm", cfg.lstm, task=task, seasonal=data_cfg.seasonal, seed=cfg.seed)
    log.info("fitting LSTM on %s -> %s (%d h)", index[ts], index[cs - 1], cs - ts)
    model.fit(scaled[ts:cs], covariates=None if cov is None else cov[ts:cs])

    bundle_dir = Path(bundle_dir or tempfile.mkdtemp(prefix="bundle-"))
    export_info = export_with_parity(
        model.network, bundle_dir / MODEL_FILE, L, 1 + len(cfg.covariates)
    )
    levels = [level_key(lv) for lv in cfg.interval_levels]
    card = ModelCard(
        model_name=cfg.model_name,
        created_at=datetime.now(UTC).isoformat(),
        architecture={"type": "lstm", **model.config()},
        input_window=L,
        max_horizon=H,
        covariates=list(cfg.covariates),
        units=data_cfg.units,
        series={
            "dataset": data_cfg.name,
            "target": data_cfg.target_col,
            "source": data_cfg.source,
            "respondent": data_cfg.eia.respondent if data_cfg.eia else None,
        },
        scaler=ScalerParams(mean=mean, scale=std),
        intervals=IntervalSpec(
            levels={lv: [0.0] * H for lv in levels}, default_level=level_key(cfg.default_level)
        ),
    )
    session = _session(bundle_dir / MODEL_FILE)

    # Split-conformal calibration on data the model never trained on.
    cw = rolling_windows(values, hours, cs, hs, L, H, stride=1)
    point = Predictor(card, session).predict_windows(cw.X, cw.hours)
    card.intervals = IntervalSpec(
        levels=conformal_halfwidths(np.abs(cw.Y - point), cfg.interval_levels),
        default_level=level_key(cfg.default_level),
        calibration={
            "start": index[cs].isoformat(),
            "end": index[hs - 1].isoformat(),
            "n_origins": len(cw.origins),
        },
    )
    predictor = Predictor(card, session)

    scale = seasonal_scale(values[max(0, hs - cfg.scale_reference_hours) : hs])
    holdout, hw, _ = evaluate_predictor(
        predictor,
        values,
        hours,
        hs,
        n,
        horizons=task.forecast_horizons,
        scale=scale,
        stride=cfg.eval_stride,
    )
    baselines = baseline_metrics(hw, scale, task.forecast_horizons, H)
    card.metrics = {"holdout": holdout, "baselines": baselines, "mase_scale": scale}
    card.training = {
        "data_config": cfg.data,
        "data_version": snap.version if snap else None,
        "data_sha256": snap.sha256 if snap else None,
        "train_start": index[ts].isoformat(),
        "train_end": index[cs - 1].isoformat(),
        "holdout_start": index[hs].isoformat(),
        "holdout_end": index[n - 1].isoformat(),
        "scale_reference_hours": cfg.scale_reference_hours,
        "seed": cfg.seed,
        "params": params,
        "best_epoch": model.best_epoch,
        "epochs_run": len(model.history),
        **export_info,
    }
    write_card(card, bundle_dir)
    log.info(
        "holdout MASE mean %.4f (seasonal naive %.4f, naive %.4f); coverage %s",
        holdout["mase_mean"],
        baselines["seasonal_naive"]["mase_mean"],
        baselines["naive"]["mase_mean"],
        {lv: round(holdout[f"coverage_{lv}"], 3) for lv in levels},
    )

    result = ProductionResult(
        bundle_dir=bundle_dir, card=card, holdout=holdout, baselines=baselines
    )
    if tracker is not None:
        _log_to_mlflow(tracker, cfg, result, model, snap, series, register, extra_tags or {})
    return result


def _log_to_mlflow(tracker, cfg, result, model, snap, series, register, extra_tags) -> None:
    import mlflow

    from forecast.train.tracking import data_tags

    card = result.card
    tags = {
        "kind": "production",
        "model_name": cfg.model_name,
        "data.config": cfg.data,
        "holdout.start": card.training["holdout_start"],
        "holdout.end": card.training["holdout_end"],
        **(data_tags(snap) if snap else {}),
        **extra_tags,
    }
    run_name = f"{cfg.model_name}@{card.training['holdout_end'][:13]}"
    with tracker.run(cfg.experiment, run_name, tags) as run:
        result.run_id = run.info.run_id
        params: dict[str, Any] = {f"lstm.{k}": v for k, v in card.training["params"].items()}
        params.update(
            {
                "seed": cfg.seed,
                "covariates": list(cfg.covariates),
                "holdout_hours": cfg.holdout_hours,
                "calibration_hours": cfg.calibration_hours,
                "train_window_hours": cfg.train_window_hours,
                "interval_levels": list(cfg.interval_levels),
                "promotion.metric": cfg.promotion.metric,
                "promotion.must_beat_baseline": cfg.promotion.must_beat_baseline,
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics({f"holdout.{k}": v for k, v in result.holdout.items()})
        for kind, m in result.baselines.items():
            mlflow.log_metrics({f"baseline.{kind}.{k}": v for k, v in m.items()})
        for row in model.history:
            mlflow.log_metrics(
                {"train_loss": row["train_loss"], "val_loss": row["val_loss"]},
                step=int(row["epoch"]),
            )
        if snap is not None:
            tracker.log_dataset(snap, series.to_frame())

        card.run_id = result.run_id
        write_card(card, result.bundle_dir)
        mlflow.log_artifacts(str(result.bundle_dir), artifact_path="bundle")
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / "lstm.pt")
            (Path(tmp) / "history.json").write_text(json.dumps(model.history))
            mlflow.log_artifacts(tmp, artifact_path="torch")

        if register:
            # The bundle is not an MLflow "logged model", so register its artefact
            # directory directly as the version's source.
            client = mlflow.MlflowClient()
            try:
                client.get_registered_model(cfg.model_name)
            except mlflow.exceptions.MlflowException:
                client.create_registered_model(
                    cfg.model_name, description="LSTM load forecaster (ONNX bundle)"
                )
            mv = client.create_model_version(
                cfg.model_name, source=mlflow.get_artifact_uri("bundle"), run_id=result.run_id
            )
            result.model_version = str(mv.version)
            for k in ("holdout.start", "holdout.end", "data.version", "data.sha256"):
                if k in tags:
                    client.set_model_version_tag(cfg.model_name, mv.version, k, tags[k])
            client.set_model_version_tag(
                cfg.model_name,
                mv.version,
                "holdout.mase_mean",
                f"{result.holdout['mase_mean']:.6f}",
            )
            log.info("registered %s version %s (run %s)", cfg.model_name, mv.version, result.run_id)
