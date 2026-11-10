# Load forecasting service: LSTM vs baselines, from notebook study to MLOps system

Hourly electricity-load forecasting, grown from a leakage-audited research study
into a system: a reproducible training package, MLflow experiment tracking with
data-version hashes, a model registry guarded by a champion/challenger promotion
gate, and an ONNX Runtime API with split-conformal prediction intervals that runs
unchanged in Docker and on AWS Lambda.

The model is deliberately small (a 32-unit LSTM). The focus is the system around it.

| | |
|---|---|
| Benchmark data | UCI ElectricityLoadDiagrams20112014, client MT_001, hourly (2012-2014) |
| Production data | EIA-930 hourly demand, PJM balancing authority (live feed, API v2) |
| Task | direct multi-horizon: 168 h of history -> next 24 h; scored at h = 1, 6, 24 |
| Headline metric | MASE vs in-sample 24 h seasonal naive (< 1 beats "same hour yesterday") |
| Serving | FastAPI + ONNX Runtime (no torch in the image), 80% / 95% conformal intervals |

## Results

Full tables and the evidence behind every claim: **[RESULTS.md](RESULTS.md)**.

- **The refactor reproduces the original study bit-for-bit wherever the committed
  code can**: 18 of 27 published (model, horizon) results match exactly, to 0.0000 on
  every fold. The other 9 (later GBM rows, 5-fold SARIMA, and the LSTM) cannot be
  produced by *any* committed code. Running the original code from git history
  side by side gives the same numbers as the refactor, so the CSVs, not the
  refactor, are the problem. The LSTM table mixed two architectures across seeds.
- **Three methodology fixes** in an apples-to-apples `aligned` benchmark: one fold
  scheme for every model; a GBM train/serve calendar skew fixed (it served every
  forecast as "hour 0"); LSTM/SARIMA no longer averaged across different folds.
- **Under one protocol the LSTM does not earn its complexity.** It is worse than
  seasonal naive at every horizon (it is competitive only on folds with plenty of
  data and a stable regime). The original "LSTM beats seasonal naive" verdict came
  from mixed configurations and fold schemes.
- **Leakage canaries with a positive control**: a shuffled future-target covariate
  must not help, while the unshuffled one must be detected as a leak.

Aligned benchmark, mean MASE over the same 5 rolling-origin folds:

| Model | h=1 | h=6 | h=24 |
|---|---|---|---|
| naive (persistence) | **0.473** | **0.644** | 0.812 |
| seasonal naive | 0.741 | 0.741 | **0.743** |
| SARIMA | 0.697 | 0.887 | 1.116 |
| GBM, original features | 0.589 | 0.866 | 1.238 |
| GBM, fixed features | 0.561 | 0.732 | 0.979 |
| LSTM (3 seeds) | 1.028 | 1.086 | 1.190 |

## Quickstart

Linux, macOS or **WSL2** (native Windows on ARM64 cannot install LightGBM/pyarrow wheels).

```bash
uv sync --all-extras          # Python 3.12, locked dependencies (uv.lock)
make test                     # fast unit tests; `make test-all` adds the model-training canaries

# M1 - reproduce the original study (downloads the 700 MB UCI file once, ~35 min on 8 cores)
make train && make evaluate   # benchmark `reference`; fails if any non-documented result differs
make benchmark                # `aligned`: every model on the same 5 folds

# M2 - production model on live EIA data (needs EIA_API_KEY in .env, see .env.example)
make data-eia train-production promote export-model
make mlflow-ui                # runs, metrics, data-version tags and the registry at :5000

# M3 - serve it
make docker-build docker-run  # http://localhost:8080/docs
make test-integration         # contract tests against the running container
make latency                  # PyTorch vs ONNX Runtime CPU latency -> experiments/results/
```

```bash
curl -s localhost:8080/forecast -H 'content-type: application/json' -d @- <<'EOF' | jq '.forecasts[0]'
{"history": [ ...168 hourly values, oldest first... ], "last_timestamp": "2026-11-09T23:00:00Z", "horizon": 24}
EOF
```

## How it works

**Data versioning.** `forecast data <name>` writes immutable snapshots to
`data/processed/<dataset>/<YYYY-MM-DDTHHMMSSZ>/` with a manifest. The manifest holds a
SHA-256 of the canonical content (UTC nanosecond timestamps plus float64 values,
stable across pandas/pyarrow versions). Every load re-verifies the hash, and every
MLflow run logs the dated prefix and the hash.

**Backtesting.** Rolling-origin expanding folds with a purge gap of input window +
max horizon (192 h), scalers fit on each training fold only, and non-overlapping test
windows per horizon. `forecast train` saves raw forecasts, and `forecast evaluate`
scores them. Keeping the two apart means scoring can be audited or extended without
refitting.

**Production training** (`configs/train/production.yaml`):

```
[ train .................................. | calibration 14 d | holdout 14 d ]
  scaler + LSTM fit                          conformal widths    metrics, promotion
```

The fitted network is exported to ONNX (dynamo exporter) and must match PyTorch
within 1e-4, or the run fails. Interval half-widths are split-conformal quantiles
per step ahead. The bundle (`model.onnx` + `metadata.json` model card) is logged to
MLflow and registered as a new version. The registered artefact is the one that was
evaluated.

**Promotion gate** (`forecast promote`): the candidate gets the `production` alias
only if, on its own holdout, it beats seasonal naive **and** the current production
version. Both bundles are scored by the same serving code on the same windows with a
shared MASE denominator. The old champion keeps a `previous` alias for rollback.

**Serving.** `src/forecast/serve/` depends on numpy, onnxruntime and FastAPI only.
`POST /forecast` returns point forecasts with 80% / 95% intervals for 1-24 h and
logs each prediction, inputs included, to a JSONL directory or S3 for monitoring.
`GET /health` reports the loaded model version, and `GET /metrics` exposes
Prometheus metrics. The request/response schema is pinned by an OpenAPI snapshot
(`tests/contract/`), and CI checks live container responses against it.

## Repository layout

```
src/forecast/
  data/       UCI + EIA ingestion, cleaning, data contract, versioned snapshots
  features/   supervised windows, calendar encodings (numpy-only twin for serving)
  models/     naive, seasonal naive, SARIMA, LightGBM (legacy + fixed), LSTM
  evaluate/   folds, backtest harness, metrics, reproduction report, holdout, latency
  train/      benchmark runner, MLflow tracking, production pipeline, conformal, ONNX export, promotion
  serve/      model bundle, ONNX predictor, FastAPI app, prediction log, metrics
configs/      data / model / train YAML (pydantic-validated: typos fail loudly)
tests/        unit (incl. leakage canaries), contract (OpenAPI), integration (container)
docker/       Dockerfile.serve (also the Lambda image, via Lambda Web Adapter)
experiments/  reference CSVs from the original study, reproduced + aligned results, plots
```

## Limitations

- One UCI client and one balancing authority; no weather or price covariates. The
  LSTM is intentionally CPU-sized and untuned.
- Split-conformal coverage assumes exchangeable residuals, which hourly load is not.
  Coverage is therefore measured on the holdout (and in production), not assumed.
- Nine published numbers from the original study cannot be reproduced from any
  committed code; they are kept as documented known differences, not silently replaced.
