"""Score all baselines through the rolling-origin backtest harness (SPEC §6, §8.1)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from eval.backtest import run_backtest, save_results
from models.naive import NaiveForecaster
from models.seasonal_naive import SeasonalNaiveForecaster


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "electricity_hourly.parquet"
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run data/download.py and data/preprocess.py first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows ({df.index[0]} to {df.index[-1]})")
    print()

    all_results = []

    models = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(seasonal_period=24),
    ]

    # Import heavier models only if available
    try:
        from models.sarima import SARIMAForecaster
        models.append(SARIMAForecaster(seasonal_period=24, max_fit_seconds=300))
    except ImportError as e:
        print(f"Skipping SARIMA: {e}")

    try:
        from models.gbm import GBMForecaster
        models.append(GBMForecaster())
    except ImportError as e:
        print(f"Skipping GBM: {e}")

    for model in models:
        print(f"{'='*60}")
        print(f"Running backtest: {model.name}")
        print(f"{'='*60}")
        start = time.time()
        results = run_backtest(model, df)
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")
        print()

        # Print per-horizon summary
        for h in sorted(results["horizon"].unique()):
            subset = results[results["horizon"] == h]
            print(f"  h={h:2d}  MAE={subset['mae'].mean():.4f}  "
                  f"RMSE={subset['rmse'].mean():.4f}  "
                  f"MASE={subset['mase'].mean():.4f}")
        print()
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    save_results(combined, "baseline_results.csv")

    # Print aggregate summary
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean across folds)")
    print(f"{'='*60}")
    summary = combined.groupby(["model", "horizon"])[["mae", "rmse", "mase"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
