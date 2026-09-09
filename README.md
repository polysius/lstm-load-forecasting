# LSTM Time-Series Forecasting — Electricity Load Prediction

Multi-horizon electricity load forecasting with a leakage-audited pipeline, benchmarking a
sequence-to-sequence LSTM against naive, seasonal-naive, SARIMA, and gradient-boosting baselines
at 1-, 6-, and 24-step hourly horizons.

## Task definition

- **Dataset:** UCI ElectricityLoadDiagrams20112014 (single client, hourly)
- **Input window:** 168 hours (one week)
- **Forecast horizons:** h = 1, 6, 24 hours ahead
- **Mode:** Direct multi-horizon — one model emits the full horizon vector

## Quickstart

```bash
# Clone and install
git clone <repo-url>
cd lstm-load-forecasting
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Download and preprocess data
python data/download.py
python data/preprocess.py

# Validate data contract
python data/validate.py

# Run baselines (M3)
# python scripts/run_baselines.py

# Train LSTM (M4)
# python scripts/run_lstm.py

# Profile latency (M5)
# python eval/profile_latency.py
```

## Results

<!-- populated by scripts/render_results.py -->

## Did the LSTM earn its complexity?

<!-- populated by scripts/render_results.py -->

## Leakage prevention

Five guards are enforced and tested in `tests/test_leakage.py`:

1. **Chronological splitting only** — no shuffling, ever
2. **Scalers fit on training data only** — validation/test are transformed, never fit
3. **Windows don't straddle split boundaries** — boundary windows are explicitly dropped
4. **Purge gap between folds** — gap of L + max(h) = 192 timesteps between train and test
5. **No target-derived features** — no rolling means or expanding stats computed before splitting

A noise-canary test trains on pure random data and asserts no model beats naive.

## Latency

<!-- populated by scripts/render_results.py -->

## Where the model fails

<!-- populated after M4/M5 diagnostics -->

## Limitations

- Single client from the UCI dataset; results may not generalize across clients
- No exogenous features (weather, price signals) — calendar features only
- Single dataset; no cross-dataset validation
- Hyperparameter sweep bounded to 30 runs
