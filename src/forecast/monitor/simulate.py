"""Planted-shift demo: replay real demand through the live API, plant a distribution
shift, and let the monitoring job detect it, trigger retraining and run the
promotion gate - with metrics streaming to Prometheus/Grafana.

Timeline over the last ``days`` of the EIA snapshot (``T0`` = start)::

    T0 ............ T0 + shift_day ........................ end
    v1 trained      load x shift_factor from here on (planted)
    on data < T0

Every simulated hour the history up to that hour is POSTed to the API (which logs
the prediction). Every simulated day the monitor scores drift and rolling error on
what has been observed so far; a debounced breach retrains on the observed data,
and a promoted candidate is exported and the API container restarted with it.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from forecast.config import (
    configs_dir,
    data_dir,
    load_data_config,
    load_monitor_config,
    load_production_config,
    project_root,
    read_yaml,
)
from forecast.data.datasets import load_series
from forecast.monitor.job import append_history, read_history, run_monitor
from forecast.monitor.sinks import FanOut, JsonLinesSink, PushgatewaySink
from forecast.monitor.traffic import post_forecast
from forecast.serve.bundle import read_card
from forecast.serve.prediction_log import read_jsonl

log = logging.getLogger(__name__)


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    production: str = "demo"
    monitor: str = "monitor"
    days: int = 42
    warmup_days: int = 7  # predictions needed before the first monitor check
    shift_day: int = 21
    shift_factor: float = 1.25
    api_url: str = "http://localhost:8080"
    pushgateway_url: str | None = "localhost:9091"
    seconds_per_hour: float = 0.05
    out_dir: str = "var/demo"
    restart_command: str = "docker compose -f monitoring/docker-compose.yml restart api"
    series_label: str | None = None  # overrides the monitor config's series name


def _restart_args(sim: SimulationConfig) -> list[str]:
    """The restart command; ``$DOCKER_BIN`` replaces ``docker`` (``docker.exe`` in WSL)."""
    args = shlex.split(sim.restart_command)
    if args and args[0] == "docker":
        args[0] = os.environ.get("DOCKER_BIN", "docker")
    return args


@dataclass
class Event:
    sim_day: float
    at: str
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


def load_simulation_config(name: str = "simulate") -> SimulationConfig:
    """``configs/<name>.yaml``, or an explicit path to a YAML file."""
    if name.endswith((".yaml", ".yml")):
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(path)
    else:
        path = configs_dir() / f"{name}.yaml"
    return SimulationConfig.model_validate(read_yaml(path) if path.exists() else {})


def _wait_for_model(url: str, version: str, timeout: float = 90.0) -> None:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url.rstrip("/") + "/health", timeout=2)
            if r.ok and r.json().get("model_version") == version:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"API did not come up with model v{version}")


def run_simulation(sim: SimulationConfig) -> dict[str, Any]:
    from forecast.train.production import train_production
    from forecast.train.promote import export_bundle, promote
    from forecast.train.tracking import Tracker

    root = project_root()
    out = root / sim.out_dir
    serving, predictions = out / "serving", out / "predictions"
    history_path, metrics_path = out / "monitor_history.jsonl", out / "metrics.jsonl"
    for p in (predictions, history_path, metrics_path):
        if p.is_dir():
            for f in p.rglob("*.jsonl"):
                f.unlink()
        elif p.exists():
            p.unlink()

    prod_cfg = load_production_config(sim.production)
    mon_cfg = load_monitor_config(sim.monitor)
    if sim.series_label:
        mon_cfg = mon_cfg.model_copy(update={"series": sim.series_label})
    data_cfg = load_data_config(prod_cfg.data)
    series, snap = load_series(data_cfg, data_dir())
    series = series.tz_convert("UTC")  # sources differ (UCI is Europe/Lisbon); replay in UTC
    end = series.index[-1].floor("D")
    t0 = end - pd.Timedelta(days=sim.days)
    shift_at = t0 + pd.Timedelta(days=sim.shift_day)
    observed = series.copy()
    observed[observed.index >= shift_at] *= sim.shift_factor  # the planted shift

    # A fresh registry per run keeps the demo reproducible and out of the main store.
    registry = out / "mlflow"
    shutil.rmtree(registry, ignore_errors=True)
    registry.mkdir(parents=True)
    tracker = Tracker(
        tracking_uri=f"sqlite:///{registry / 'mlflow.db'}",
        artifact_root=str(registry / "artifacts"),
    )
    sinks = FanOut(
        JsonLinesSink(metrics_path),
        *([PushgatewaySink(sim.pushgateway_url)] if sim.pushgateway_url else []),
    )
    events: list[Event] = []

    def sim_day(t: pd.Timestamp) -> float:
        return round((t - t0) / pd.Timedelta(days=1), 3)

    def deploy(version: str, t: pd.Timestamp, why: str) -> None:
        # Copy into the existing directory: it is bind-mounted into the API
        # container, and replacing the directory itself can strand the mount.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_bundle(
                prod_cfg.model_name, Path(tmp) / "bundle", version=version, tracker=tracker
            )
            serving.mkdir(parents=True, exist_ok=True)
            for f in bundle.iterdir():
                shutil.copy2(f, serving / f.name)
        subprocess.run(_restart_args(sim), cwd=root, check=True, capture_output=True)
        _wait_for_model(sim.api_url, version)
        events.append(
            Event(sim_day(t), t.isoformat(), "deployed", {"version": version, "why": why})
        )
        log.info("deployed model v%s (%s)", version, why)

    def train_and_gate(t: pd.Timestamp, first: bool) -> None:
        tag = {"simulation": "planted-shift", "sim_day": str(sim_day(t))}
        result = train_production(
            prod_cfg, series=observed, until=t, tracker=tracker, extra_tags=tag
        )
        rule = prod_cfg.promotion
        if first:
            rule = rule.model_copy(update={"must_beat_baseline": None})
        decision = promote(
            prod_cfg.model_name,
            rule=rule,
            version=result.model_version,
            tracker=tracker,
            series=observed,
        )
        events.append(
            Event(
                sim_day(t),
                t.isoformat(),
                "promoted" if decision.promote else "rejected",
                {
                    "version": result.model_version,
                    "candidate": decision.candidate,
                    "production": decision.production,
                    "baseline": decision.baseline,
                    "reasons": decision.reasons,
                },
            )
        )
        if decision.promote:
            deploy(result.model_version, t, "initial" if first else "retrained after drift")

    log.info(
        "simulation %s -> %s, shift x%.2f at %s (dataset %s@%s)",
        t0,
        end,
        sim.shift_factor,
        shift_at,
        snap.dataset,
        snap.version,
    )
    events.append(
        Event(
            0,
            t0.isoformat(),
            "start",
            {"shift_at": shift_at.isoformat(), "shift_factor": sim.shift_factor},
        )
    )
    train_and_gate(t0, first=True)

    import requests

    session = requests.Session()
    L = data_cfg.task.input_window
    hours = pd.date_range(t0, end - pd.Timedelta(hours=24), freq="1h", tz="UTC")
    for t in hours:
        if t == shift_at:
            events.append(
                Event(sim_day(t), t.isoformat(), "shift_planted", {"factor": sim.shift_factor})
            )
        window = observed[observed.index <= t].iloc[-L:]
        if len(window) == L and not window.isna().any():
            try:
                post_forecast(
                    sim.api_url,
                    window.round(3).tolist(),
                    t.to_pydatetime(),
                    series_id=mon_cfg.series,
                    session=session,
                )
            except requests.RequestException as exc:  # e.g. during a restart
                log.warning("request at %s failed: %s", t, exc)
        time.sleep(sim.seconds_per_hour)

        if t.hour == 0 and t >= t0 + pd.Timedelta(days=sim.warmup_days):
            card = read_card(serving)
            result = run_monitor(
                mon_cfg,
                actuals=observed[observed.index <= t],
                records=read_jsonl(predictions),
                card=card,
                history=read_history(history_path),
                as_of=t,
            )
            append_history(history_path, result)
            sinks.publish(
                {**result.metrics(), "SimulatedDay": sim_day(t)},
                series=mon_cfg.series,
                timestamp=t.to_pydatetime(),
            )
            events.append(
                Event(
                    sim_day(t),
                    t.isoformat(),
                    "monitor",
                    {
                        "psi": result.drift["psi"],
                        "threshold": result.threshold,
                        "mase_trigger": result.mase_trigger,
                        "breach": result.breach,
                        "retrain": result.retrain,
                        **result.performance,
                    },
                )
            )
            if result.retrain:
                events.append(
                    Event(
                        sim_day(t), t.isoformat(), "retrain_triggered", {"reasons": result.reasons}
                    )
                )
                train_and_gate(t, first=False)

    summary = {
        "dataset": f"{snap.dataset}@{snap.version}",
        "start": t0.isoformat(),
        "end": end.isoformat(),
        "shift_at": shift_at.isoformat(),
        "shift_factor": sim.shift_factor,
        "config": sim.model_dump(),
        "events": [asdict(e) for e in events],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    try:
        from forecast.monitor.plots import plot_simulation

        plot_simulation(summary, out / "planted_shift.png")
    except Exception:
        log.exception("plotting failed")
    return summary
