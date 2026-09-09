"""Train and evaluate LSTM through the rolling-origin backtest harness (SPEC §7, §8.1).

Reports results as mean +/- std over multiple seeds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from eval.backtest import run_backtest, save_results
from models.lstm import LSTMForecaster

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_path = _ROOT / "data" / "processed" / "electricity_hourly.parquet"
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    with open(_ROOT / "config" / "lstm.yaml") as f:
        cfg = yaml.safe_load(f)

    seeds = cfg["training"]["seeds"]
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows ({df.index[0]} to {df.index[-1]})")
    print(f"Seeds: {seeds}")
    print()

    all_results = []

    for seed in seeds:
        print(f"{'='*60}")
        print(f"LSTM seed={seed}")
        print(f"{'='*60}")

        model = LSTMForecaster(seed=seed)
        start = time.time()
        results = run_backtest(model, df)
        elapsed = time.time() - start
        results["seed"] = seed

        print(f"\n  Completed in {elapsed:.1f}s")
        for h in sorted(results["horizon"].unique()):
            s = results[results["horizon"] == h]
            print(f"  h={h:2d}  MAE={s['mae'].mean():.4f}  MASE={s['mase'].mean():.4f}")
        print()
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    save_results(combined, "lstm_results.csv")

    # Summary: mean +/- std across seeds
    print(f"\n{'='*60}")
    print("LSTM RESULTS: mean +/- std across seeds")
    print(f"{'='*60}")
    for h in sorted(combined["horizon"].unique()):
        subset = combined[combined["horizon"] == h]
        seed_means = subset.groupby("seed")["mase"].mean()
        print(f"  h={h:2d}  MASE = {seed_means.mean():.4f} +/- {seed_means.std():.4f}")


if __name__ == "__main__":
    main()
