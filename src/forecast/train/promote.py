"""Champion/challenger promotion against the MLflow model registry.

A candidate version gets the ``production`` alias only if, on the candidate's own
holdout window, it (a) beats the configured baseline and (b) beats the current
production version by at least ``min_improvement``. Both bundles are scored by
the same serving code on the same windows with the same MASE denominator. The
previous champion keeps a ``previous`` alias for one-command rollback.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import PromotionRule, data_dir, load_data_config
from forecast.data.datasets import load_series
from forecast.evaluate.holdout import baseline_metrics, evaluate_predictor, seasonal_scale
from forecast.features import calendar_np
from forecast.serve.bundle import read_card
from forecast.serve.predictor import Predictor

log = logging.getLogger(__name__)

PRODUCTION_ALIAS = "production"
PREVIOUS_ALIAS = "previous"
# Error codes MLflow stores raise when a model has no version under an alias.
_ALIAS_NOT_FOUND = frozenset({"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"})


@dataclass
class Decision:
    promote: bool
    model_name: str
    candidate_version: str
    production_version: str | None
    metric: str
    candidate: float
    production: float | None
    baseline: float | None
    reasons: list[str] = field(default_factory=list)
    window: dict[str, str] = field(default_factory=dict)
    applied: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def decide(
    candidate: float, production: float | None, baseline: float | None, rule: PromotionRule
) -> tuple[bool, list[str]]:
    """Pure rule: lower metric is better. Returns (promote?, human-readable reasons)."""
    if not math.isfinite(candidate):
        return False, [f"candidate {rule.metric} is not finite"]
    ok, reasons = True, []
    if rule.must_beat_baseline and baseline is not None:
        if candidate < baseline:
            reasons.append(f"beats {rule.must_beat_baseline} ({candidate:.4f} < {baseline:.4f})")
        else:
            ok = False
            reasons.append(
                f"does not beat {rule.must_beat_baseline} ({candidate:.4f} >= {baseline:.4f})"
            )
    if production is None:
        reasons.append("no current production model")
    elif candidate < production - rule.min_improvement:
        reasons.append(
            f"beats production ({candidate:.4f} < {production:.4f} - {rule.min_improvement})"
        )
    else:
        ok = False
        reasons.append(
            f"does not beat production ({candidate:.4f} >= {production:.4f} "
            f"- {rule.min_improvement})"
        )
    return ok, reasons


def _download_bundle(tracking_uri: str, run_id: str, dst: Path) -> Path:
    import mlflow

    dst.mkdir(parents=True, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="bundle", dst_path=str(dst), tracking_uri=tracking_uri
    )
    return Path(path)


def _latest_version(client, name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise LookupError(f"no registered versions for {name!r}")
    return str(max(int(v.version) for v in versions))


def score_bundle(
    bundle: Path,
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    scale_reference_hours: int,
    horizons,
) -> tuple[dict, dict]:
    """Score one bundle on ``[start, end]``; also the baselines on the same windows."""
    predictor = Predictor.from_dir(bundle)
    values = series.to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(series.index)
    hours = calendar_np.epoch_hours(index)
    hs = int(index.searchsorted(start))
    he = int(index.searchsorted(end, side="right"))
    scale = seasonal_scale(values[max(0, hs - scale_reference_hours) : hs])
    metrics, w, _ = evaluate_predictor(
        predictor, values, hours, hs, he, horizons=horizons, scale=scale
    )
    return metrics, baseline_metrics(w, scale, horizons, predictor.max_horizon)


def promote(
    model_name: str,
    *,
    rule: PromotionRule,
    version: str | None = None,
    tracker=None,
    dry_run: bool = False,
    series: pd.Series | None = None,
) -> Decision:
    import mlflow
    from mlflow.exceptions import MlflowException

    from forecast.train.tracking import Tracker

    tracker = tracker or Tracker.from_env()
    mlflow.set_tracking_uri(tracker.tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracker.tracking_uri)
    version = version or _latest_version(client, model_name)
    cand = client.get_model_version(model_name, version)
    run = client.get_run(cand.run_id)
    tags = run.data.tags
    start, end = pd.Timestamp(tags["holdout.start"]), pd.Timestamp(tags["holdout.end"])

    if series is None:
        data_cfg = load_data_config(tags["data.config"])
        series, _ = load_series(data_cfg, data_dir(), version=tags.get("data.version"))
        horizons = data_cfg.task.forecast_horizons
    else:
        horizons = (1, 6, 24)

    prod = None
    try:
        prod = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException as exc:
        if exc.error_code not in _ALIAS_NOT_FOUND:
            raise
    if prod is not None and str(prod.version) == str(version):
        raise ValueError(f"version {version} is already {PRODUCTION_ALIAS}")

    with tempfile.TemporaryDirectory() as tmp:
        cand_dir = _download_bundle(tracker.tracking_uri, cand.run_id, Path(tmp) / "candidate")
        ref_hours = int(read_card(cand_dir).training.get("scale_reference_hours", 672))
        cand_m, base_m = score_bundle(
            cand_dir, series, start, end, scale_reference_hours=ref_hours, horizons=horizons
        )
        prod_m = None
        if prod is not None:
            prod_dir = _download_bundle(tracker.tracking_uri, prod.run_id, Path(tmp) / "production")
            prod_m, _ = score_bundle(
                prod_dir, series, start, end, scale_reference_hours=ref_hours, horizons=horizons
            )

    metric = rule.metric
    baseline = base_m[rule.must_beat_baseline][metric] if rule.must_beat_baseline else None
    ok, reasons = decide(cand_m[metric], prod_m[metric] if prod_m else None, baseline, rule)
    decision = Decision(
        promote=ok,
        model_name=model_name,
        candidate_version=str(version),
        production_version=str(prod.version) if prod is not None else None,
        metric=metric,
        candidate=cand_m[metric],
        production=prod_m[metric] if prod_m else None,
        baseline=baseline,
        reasons=reasons,
        window={"start": start.isoformat(), "end": end.isoformat()},
    )

    stamp = datetime.now(UTC).isoformat()
    client.set_model_version_tag(
        model_name, version, "promotion.decision", "promoted" if ok else "rejected"
    )
    client.set_model_version_tag(model_name, version, "promotion.reasons", "; ".join(reasons))
    client.set_model_version_tag(
        model_name, version, "promotion.compared_to", decision.production_version or "none"
    )
    client.set_model_version_tag(model_name, version, "promotion.evaluated_at", stamp)
    if ok and not dry_run:
        if prod is not None:
            client.set_registered_model_alias(model_name, PREVIOUS_ALIAS, prod.version)
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)
        decision.applied = True
    log.info(
        "%s v%s: %s (%s)", model_name, version, "PROMOTED" if ok else "rejected", "; ".join(reasons)
    )
    return decision


def export_bundle(
    model_name: str,
    out_dir: Path,
    *,
    alias: str | None = PRODUCTION_ALIAS,
    version: str | None = None,
    tracker=None,
) -> Path:
    """Download a registered bundle for serving and stamp its version into the card."""
    import shutil

    import mlflow

    from forecast.serve.bundle import write_card
    from forecast.train.tracking import Tracker

    tracker = tracker or Tracker.from_env()
    client = mlflow.MlflowClient(tracking_uri=tracker.tracking_uri)
    mv = (
        client.get_model_version(model_name, version)
        if version
        else client.get_model_version_by_alias(model_name, alias)
    )
    with tempfile.TemporaryDirectory() as tmp:
        src = _download_bundle(tracker.tracking_uri, mv.run_id, Path(tmp))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(src, out_dir)
    card = read_card(out_dir)
    card.model_version = str(mv.version)
    card.run_id = mv.run_id
    write_card(card, out_dir)
    return out_dir
