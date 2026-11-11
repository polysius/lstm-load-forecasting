"""Local retraining: check drift, and when it is confirmed, retrain and run the promotion gate.

This is the on-demand equivalent of a scheduled retraining job. Run it by hand, from
cron or from a systemd timer; it needs only the prediction log, the EIA snapshot and
the MLflow store on this machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from forecast.config import (
    MonitorConfig,
    data_dir,
    load_data_config,
    load_production_config,
)
from forecast.data.datasets import load_series
from forecast.monitor.job import (
    MonitorResult,
    append_history,
    load_prediction_records,
    read_history,
    run_monitor,
)
from forecast.serve.bundle import read_card

log = logging.getLogger(__name__)


@dataclass
class RetrainOutcome:
    monitor: MonitorResult
    retrained: bool
    promoted: bool | None = None
    model_version: str | None = None


def check_drift(
    cfg: MonitorConfig,
    *,
    predictions: Path,
    history_path: Path,
    model_dir: Path,
    report: Path | None = None,
) -> MonitorResult:
    """Score drift and rolling error for the newest data; ``result.retrain`` is only a recommendation."""
    actuals, _ = load_series(load_data_config(cfg.data), data_dir())
    card = read_card(model_dir)
    since = actuals.index[-1] - pd.Timedelta(hours=2 * cfg.window_hours)
    records = load_prediction_records(str(predictions), since=since)
    result = run_monitor(
        cfg,
        actuals=actuals,
        records=records,
        card=card,
        history=read_history(history_path),
        html=report,
    )
    return result


def record_check(history_path: Path, result: MonitorResult, *, retrained: bool) -> None:
    """Append a check to the history; its ``retrain`` flag says a retrain happened, not that one was advised."""
    append_history(history_path, replace(result, retrain=retrained))


def retrain_if_drifted(
    cfg: MonitorConfig,
    *,
    predictions: Path,
    history_path: Path,
    model_dir: Path,
    report: Path | None = None,
    force: bool = False,
) -> RetrainOutcome:
    """Run the drift check; retrain and apply the promotion gate when it says so (or ``force``)."""
    result = check_drift(
        cfg, predictions=predictions, history_path=history_path, model_dir=model_dir, report=report
    )
    if not (result.retrain or force):
        record_check(history_path, result, retrained=False)
        return RetrainOutcome(monitor=result, retrained=False)

    from forecast.train.production import train_production
    from forecast.train.promote import export_bundle, promote
    from forecast.train.tracking import Tracker

    production = load_production_config(cfg.production)
    trained = train_production(production, tracker=Tracker.from_env())
    record_check(history_path, result, retrained=True)
    decision = promote(production.model_name, rule=production.promotion)
    if decision.promote:
        export_bundle(production.model_name, model_dir, alias="production")
        log.info("promoted v%s; bundle refreshed in %s", trained.model_version, model_dir)
    else:
        log.info("v%s not promoted: %s", trained.model_version, "; ".join(decision.reasons))
    return RetrainOutcome(
        monitor=result,
        retrained=True,
        promoted=decision.promote,
        model_version=trained.model_version,
    )
