"""``forecast`` command line: one entry point for data, training, evaluation and ops."""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "botocore", "matplotlib", "alembic", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_data(args: argparse.Namespace) -> int:
    from forecast.config import data_dir, load_data_config
    from forecast.data.datasets import build_dataset

    cfg = load_data_config(args.dataset)
    snap = build_dataset(cfg, data_dir(), force_download=args.force_download)
    print(f"{snap.dataset} {snap.version} rows={snap.rows} sha256={snap.sha256}")
    print(f"  {snap.start} -> {snap.end}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from forecast.config import load_benchmark_config
    from forecast.train.benchmark import train_benchmark

    cfg = load_benchmark_config(args.benchmark)
    tracker = None
    if not args.no_mlflow:
        from forecast.train.tracking import Tracker

        tracker = Tracker.from_env()
    train_benchmark(
        cfg, runs=args.run, models=args.model, skip_existing=args.skip_existing, tracker=tracker
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from forecast.config import load_benchmark_config
    from forecast.evaluate.report import evaluate_benchmark, render_summary

    cfg = load_benchmark_config(args.benchmark)
    report = evaluate_benchmark(cfg)
    print(render_summary(cfg, report))
    if report.missing:
        print(
            f"missing forecasts for {len(report.missing)} jobs; run `forecast train` first",
            file=sys.stderr,
        )
        return 2
    if not report.reproduced and not args.no_check:
        print("reproduction check FAILED", file=sys.stderr)
        return 1
    return 0


def cmd_train_production(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from forecast.config import load_production_config
    from forecast.train.production import train_production
    from forecast.train.tracking import Tracker

    cfg = load_production_config(args.config)
    result = train_production(
        cfg,
        data_version=args.data_version,
        until=args.until,
        tracker=None if args.no_mlflow else Tracker.from_env(),
        bundle_dir=Path(args.bundle_dir) if args.bundle_dir else None,
        register=not args.no_register,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "model_version": result.model_version,
                "bundle_dir": str(result.bundle_dir),
                "holdout_mase_mean": result.holdout["mase_mean"],
                "seasonal_naive_mase_mean": result.baselines["seasonal_naive"]["mase_mean"],
            },
            indent=2,
        )
    )
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from forecast.config import load_production_config
    from forecast.train.promote import promote

    cfg = load_production_config(args.config)
    decision = promote(
        cfg.model_name, rule=cfg.promotion, version=args.version, dry_run=args.dry_run
    )
    print(decision.to_json())
    return 3 if (args.fail_on_reject and not decision.promote) else 0


def cmd_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from forecast.config import load_production_config
    from forecast.serve.bundle import read_card
    from forecast.train.promote import export_bundle

    cfg = load_production_config(args.config)
    out = export_bundle(cfg.model_name, Path(args.out), alias=args.alias, version=args.version)
    card = read_card(out)
    print(f"exported {card.model_name} v{card.model_version} (run {card.run_id}) -> {out}")
    return 0


def cmd_smoke_bundle(args: argparse.Namespace) -> int:
    from pathlib import Path

    from forecast.train.smoke import build_smoke_bundle

    out = build_smoke_bundle(Path(args.out), epochs=args.epochs)
    print(f"smoke bundle -> {out}")
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    from pathlib import Path

    from forecast.serve.app import create_app
    from forecast.serve.contract import write_snapshot

    print(f"wrote {write_snapshot(create_app(), Path(args.out))}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    os.environ.setdefault("MODEL_DIR", args.model_dir)
    if args.prediction_log:
        os.environ["PREDICTION_LOG_URI"] = args.prediction_log
    uvicorn.run("forecast.serve.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def cmd_bench_latency(args: argparse.Namespace) -> int:
    from pathlib import Path

    from forecast.evaluate.latency import benchmark, to_markdown

    kwargs = {}
    if args.bundle:
        from forecast.serve.bundle import read_card

        card = read_card(Path(args.bundle))
        kwargs = {
            "hidden_size": card.architecture["hidden_size"],
            "num_layers": card.architecture["num_layers"],
            "input_window": card.input_window,
            "max_horizon": card.max_horizon,
        }
    df = benchmark(runs=args.runs, warmup=args.warmup, **kwargs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(to_markdown(df))
    print(f"\nwrote {out}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime
    from pathlib import Path

    from forecast.config import load_monitor_config
    from forecast.monitor.retrain import check_drift

    cfg = load_monitor_config(args.config)
    result = check_drift(
        cfg,
        predictions=Path(args.predictions),
        history_path=Path(args.history),
        model_dir=Path(args.model_dir),
        report=Path(args.report) if args.report else None,
    )
    if args.pushgateway:
        from forecast.monitor.sinks import PushgatewaySink

        PushgatewaySink(args.pushgateway).publish(
            result.metrics(), series=cfg.series, timestamp=datetime.now(UTC)
        )
    print(result.to_json())
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict
    from pathlib import Path

    from forecast.config import load_monitor_config
    from forecast.monitor.retrain import retrain_if_drifted

    outcome = retrain_if_drifted(
        load_monitor_config(args.config),
        predictions=Path(args.predictions),
        history_path=Path(args.history),
        model_dir=Path(args.model_dir),
        report=Path(args.report) if args.report else None,
        force=args.force,
    )
    print(json.dumps(asdict(outcome), default=float, indent=2))
    return 0


def cmd_traffic(args: argparse.Namespace) -> int:
    import json

    from forecast.config import load_data_config
    from forecast.monitor.traffic import send_live_traffic

    print(json.dumps(send_live_traffic(load_data_config(args.data), args.url), indent=2))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    import json

    from forecast.monitor.simulate import load_simulation_config, run_simulation

    print(json.dumps(run_simulation(load_simulation_config(args.config)), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="fetch, preprocess, validate and snapshot a dataset")
    p.add_argument("dataset", help="data config name (configs/data/<name>.yaml)")
    p.add_argument("--force-download", action="store_true")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="fit every model of a benchmark over rolling-origin folds")
    p.add_argument("--benchmark", default="reference", help="configs/train/<name>.yaml")
    p.add_argument("--run", action="append", help="only this run (repeatable)")
    p.add_argument("--model", action="append", help="only this model/label (repeatable)")
    p.add_argument("--skip-existing", action="store_true", help="keep forecasts already on disk")
    p.add_argument("--no-mlflow", action="store_true", help="do not log runs to MLflow")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score forecasts and check against reference results")
    p.add_argument("--benchmark", default="reference")
    p.add_argument("--no-check", action="store_true", help="report but do not fail on mismatch")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train-production", help="train, calibrate, export and register a model")
    p.add_argument("--config", default="production", help="configs/train/<name>.yaml")
    p.add_argument("--data-version", help="pin a dataset snapshot (default: latest)")
    p.add_argument("--until", help="train as of this timestamp (simulations)")
    p.add_argument("--bundle-dir", help="write the bundle here (default: a temp dir)")
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.set_defaults(func=cmd_train_production)

    p = sub.add_parser("promote", help="champion/challenger gate for a registered version")
    p.add_argument("--config", default="production")
    p.add_argument("--version", help="candidate version (default: latest)")
    p.add_argument("--dry-run", action="store_true", help="decide and tag, but move no alias")
    p.add_argument("--fail-on-reject", action="store_true", help="exit 3 when rejected")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("export", help="download a registered bundle for serving")
    p.add_argument("--config", default="production")
    p.add_argument("--alias", default="production")
    p.add_argument("--version")
    p.add_argument("--out", default="artifacts/model")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("smoke-bundle", help="tiny synthetic bundle for tests and CI images")
    p.add_argument("--out", default="artifacts/model")
    p.add_argument("--epochs", type=int, default=3)
    p.set_defaults(func=cmd_smoke_bundle)

    p = sub.add_parser("openapi", help="write the normalised OpenAPI contract snapshot")
    p.add_argument("--out", default="tests/contract/openapi.snapshot.json")
    p.set_defaults(func=cmd_openapi)

    p = sub.add_parser("serve", help="run the API locally with uvicorn")
    p.add_argument("--model-dir", default="artifacts/model")
    p.add_argument("--prediction-log", default="var/predictions")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("bench-latency", help="PyTorch vs ONNX Runtime CPU latency")
    p.add_argument("--bundle", help="take the architecture from this bundle's card")
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--out", default="experiments/results/latency.csv")
    p.set_defaults(func=cmd_bench_latency)

    monitor_args = argparse.ArgumentParser(add_help=False)
    monitor_args.add_argument("--config", default="monitor", help="configs/<name>.yaml")
    monitor_args.add_argument("--predictions", default="var/predictions", help="prediction log dir")
    monitor_args.add_argument("--history", default="var/monitoring/history.jsonl")
    monitor_args.add_argument("--model-dir", default="artifacts/model", help="serving bundle")
    monitor_args.add_argument("--report", help="write the Evidently drift report (HTML) here")

    p = sub.add_parser(
        "monitor", parents=[monitor_args], help="score input drift and rolling forecast error"
    )
    p.add_argument("--pushgateway", help="host:port of a Prometheus Pushgateway to publish to")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser(
        "retrain",
        parents=[monitor_args],
        help="check drift; if confirmed, retrain and apply the promotion gate",
    )
    p.add_argument("--force", action="store_true", help="retrain even without a drift trigger")
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("traffic", help="send the latest live EIA week to a running API")
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--data", default="eia", help="configs/data/<name>.yaml")
    p.set_defaults(func=cmd_traffic)

    p = sub.add_parser("simulate", help="planted-shift demo: drift -> retrain -> promotion")
    p.add_argument("--config", default="simulate", help="configs/<name>.yaml")
    p.set_defaults(func=cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
