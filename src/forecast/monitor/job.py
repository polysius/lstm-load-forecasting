"""The monitoring job: drift + rolling error -> metrics -> a debounced retraining decision.

Run it on demand with ``forecast monitor`` (or ``make monitor``) against the API's
local prediction log, and inside the planted-shift simulator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.config import MonitorConfig
from forecast.monitor.drift import drift_scores, psi
from forecast.monitor.performance import join_actuals, predictions_frame, rolling_metrics
from forecast.serve.bundle import ModelCard

log = logging.getLogger(__name__)

YEAR = pd.Timedelta(days=364)  # 52 weeks: keeps weekdays aligned


@dataclass
class MonitorResult:
    as_of: str
    model_version: str
    reference: dict[str, str]
    drift: dict[str, float]
    performance: dict[str, float]
    breach: bool
    retrain: bool
    threshold: float = float("nan")
    threshold_source: str = ""
    retrains_total: int = 0
    mase_trigger: float = float("nan")
    reasons: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return {
            "DriftPSI": self.drift["psi"],
            "DriftThreshold": self.threshold,
            "DriftBreach": 1.0 if self.breach else 0.0,
            "DriftKS": self.drift["ks_statistic"],
            "RollingMASE_h1": self.performance.get("mase_h1", float("nan")),
            "RollingMASE_h24": self.performance.get("mase_h24", float("nan")),
            "Coverage80": self.performance.get("coverage", float("nan")),
            "RetrainTriggered": 1.0 if self.retrain else 0.0,
            "RetrainsTotal": float(self.retrains_total),
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=float)


def reference_window(
    cfg: MonitorConfig, as_of: pd.Timestamp, card: ModelCard
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if cfg.reference == "year_over_year":
        center = as_of - pd.Timedelta(hours=cfg.window_hours / 2) - YEAR
        half = pd.Timedelta(hours=cfg.reference_hours / 2)
        return center - half, center + half
    end = pd.Timestamp(card.training["holdout_end"])
    return end - pd.Timedelta(hours=cfg.reference_hours), end


def psi_null(
    cfg: MonitorConfig,
    actuals: pd.Series,
    *,
    offset: pd.Timedelta,
    until: pd.Timestamp,
    step_hours: int = 24,
) -> np.ndarray:
    """PSI at daily checks over history ending at ``until``: the statistic's null.

    Each historical check compares its window with a reference ending ``offset``
    earlier - the same lag as the live comparison (the model's age for the
    training-tail reference, about a year for year-over-year) - so ordinary seasonal
    divergence at that lag is part of the null. Nothing was planted in this history,
    so a high quantile is the level ordinary variation reaches.
    """
    idx = pd.DatetimeIndex(actuals.index)
    vals = actuals.to_numpy(dtype=float)
    window = pd.Timedelta(hours=cfg.window_hours)
    ref_len = pd.Timedelta(hours=cfg.reference_hours)
    out = []
    t = idx[0] + max(window, offset + ref_len)
    while t <= until:
        cur = vals[idx.searchsorted(t - window, side="right") : idx.searchsorted(t, side="right")]
        ref = vals[idx.searchsorted(t - offset - ref_len) : idx.searchsorted(t - offset)]
        if np.isfinite(ref).sum() >= 24 and np.isfinite(cur).sum() >= 24:
            out.append(psi(ref, cur, cfg.psi_bins))
        t += pd.Timedelta(hours=step_hours)
    return np.asarray(out)


def resolve_threshold(
    cfg: MonitorConfig,
    actuals: pd.Series,
    *,
    as_of: pd.Timestamp,
    ref_start: pd.Timestamp,
    ref_end: pd.Timestamp,
) -> tuple[float, str]:
    if cfg.psi_threshold != "auto":
        return float(cfg.psi_threshold), "fixed"
    null = psi_null(cfg, actuals, offset=as_of - ref_end, until=ref_start)
    if len(null) < 30:
        return cfg.psi_threshold_floor, f"floor ({len(null)} historical checks, need 30)"
    q = float(np.quantile(null, cfg.psi_threshold_quantile))
    return max(cfg.psi_threshold_floor, q), (
        f"auto: q{cfg.psi_threshold_quantile:g} of {len(null)} historical checks = {q:.3f}"
    )


def decide_retrain(
    breach: bool,
    history: list[dict[str, Any]],
    cfg: MonitorConfig,
    as_of: pd.Timestamp,
    model_version: str | None = None,
) -> tuple[bool, list[str]]:
    """Retrain after ``consecutive_breaches`` breaches in a row, outside the cooldown.

    Only checks of the currently deployed model count toward the streak: a new model
    starts with a clean slate instead of inheriting its predecessor's breaches.
    """
    if not breach:
        return False, ["no breach"]
    previous = [h for h in history if pd.Timestamp(h["as_of"]) < as_of]
    retrains = [pd.Timestamp(h["as_of"]) for h in previous if h.get("retrain")]
    if model_version is not None:
        previous = [h for h in previous if str(h.get("model_version")) == str(model_version)]
    streak = 1
    for h in reversed(previous):
        if not h.get("breach"):
            break
        streak += 1
    if streak < cfg.consecutive_breaches:
        return False, [f"breach {streak}/{cfg.consecutive_breaches} (debounce)"]
    if retrains:
        since = (as_of - max(retrains)) / pd.Timedelta(hours=1)
        if since < cfg.cooldown_hours:
            return False, [f"cooldown: last retrain {since:.0f} h ago (< {cfg.cooldown_hours} h)"]
    return True, [f"{streak} consecutive breaches"]


def run_monitor(
    cfg: MonitorConfig,
    *,
    actuals: pd.Series,
    records: list[dict[str, Any]],
    card: ModelCard,
    history: list[dict[str, Any]] | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    html: Path | None = None,
) -> MonitorResult:
    actuals = actuals.copy()
    actuals.index = pd.DatetimeIndex(actuals.index).tz_convert("UTC")
    as_of = pd.Timestamp(as_of) if as_of is not None else actuals.dropna().index[-1]
    as_of = as_of.tz_convert("UTC") if as_of.tzinfo else as_of.tz_localize("UTC")
    window = pd.Timedelta(hours=cfg.window_hours)

    current = actuals[(actuals.index > as_of - window) & (actuals.index <= as_of)]
    ref_start, ref_end = reference_window(cfg, as_of, card)
    reference = actuals[(actuals.index >= ref_start) & (actuals.index < ref_end)]
    if reference.dropna().size < 24 or current.dropna().size < 24:
        raise ValueError(f"not enough data: reference {reference.size} h, current {current.size} h")
    drift = drift_scores(reference.to_numpy(), current.to_numpy(), bins=cfg.psi_bins, html=html)

    # Score only the deployed model: its predecessor's errors are not its own.
    preds = predictions_frame(records)
    preds = preds[preds["model_version"] == str(card.model_version)]
    joined = join_actuals(preds, actuals)
    # Origins whose full 24 h horizon has been observed by as_of.
    perf = rolling_metrics(
        joined,
        start=as_of - window,
        end=as_of - pd.Timedelta(hours=23),
        scale=float(card.metrics["mase_scale"]),
    )

    threshold, source = resolve_threshold(
        cfg, actuals, as_of=as_of, ref_start=ref_start, ref_end=ref_end
    )
    reasons = []
    breach = drift.psi > threshold
    if breach:
        reasons.append(f"PSI {drift.psi:.3f} > {threshold:.3f} ({source})")
    rolling = perf.get("mase_h24", float("nan"))
    if cfg.mase_threshold is not None and rolling > cfg.mase_threshold:
        breach = True
        reasons.append(f"rolling MASE h24 {rolling:.3f} > {cfg.mase_threshold}")
    validated = float(card.metrics.get("holdout", {}).get("mase_h24", float("nan")))
    if cfg.mase_degradation is not None and rolling > cfg.mase_degradation * validated:
        breach = True
        reasons.append(
            f"rolling MASE h24 {rolling:.3f} > {cfg.mase_degradation:g} x validated {validated:.3f}"
        )
    retrain, why = decide_retrain(breach, history or [], cfg, as_of, card.model_version)
    result = MonitorResult(
        as_of=as_of.isoformat(),
        model_version=card.model_version,
        reference={
            "kind": cfg.reference,
            "start": ref_start.isoformat(),
            "end": ref_end.isoformat(),
        },
        drift=drift.to_dict(),
        performance=perf,
        breach=bool(breach),
        retrain=retrain,
        threshold=threshold,
        threshold_source=source,
        retrains_total=sum(1 for h in history or [] if h.get("retrain")) + int(retrain),
        mase_trigger=(cfg.mase_degradation * validated) if cfg.mase_degradation else float("nan"),
        reasons=reasons + why,
    )
    log.info(
        "monitor %s: PSI %.3f (threshold %.3f) KS %.3f MASE h1 %.3f h24 %.3f cov %.3f -> %s",
        as_of,
        drift.psi,
        threshold,
        drift.ks_statistic,
        perf.get("mase_h1", float("nan")),
        perf.get("mase_h24", float("nan")),
        perf.get("coverage", float("nan")),
        "RETRAIN" if retrain else ("breach" if breach else "ok"),
    )
    return result


# ------------------------------------------------------------------ state + logs IO


def read_history(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_history(path: Path, result: MonitorResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(result.to_json() + "\n")


def load_prediction_records(uri: str, *, since: pd.Timestamp) -> list[dict[str, Any]]:
    """Prediction logs from a local JSONL directory, keeping origins on or after ``since``."""
    first_day = since.strftime("%Y-%m-%d")
    from forecast.serve.prediction_log import read_jsonl

    root = Path(uri[len("file://") :] if uri.startswith("file://") else uri)
    return [r for r in read_jsonl(root) if str(r["origin"])[:10] >= first_day]
