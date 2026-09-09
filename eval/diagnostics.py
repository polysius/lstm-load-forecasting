"""Diagnostic plots: residuals, error distribution, failure analysis (SPEC §8.4)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_PLOTS_DIR = _ROOT / "experiments" / "plots"


def ensure_plots_dir() -> Path:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return _PLOTS_DIR


def plot_residuals_over_time(
    results_csv: Path,
    model_name: str = "lstm",
) -> None:
    """Plot residuals over time for a given model."""
    plots = ensure_plots_dir()
    df = pd.read_csv(results_csv)
    model_df = df[df["model"] == model_name]

    if model_df.empty:
        print(f"No results for model '{model_name}'")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, h in enumerate(sorted(model_df["horizon"].unique())):
        ax = axes[idx]
        subset = model_df[model_df["horizon"] == h]
        ax.bar(subset["fold"], subset["mae"], alpha=0.7)
        ax.set_title(f"h={h}")
        ax.set_xlabel("Fold")
        ax.set_ylabel("MAE")

    fig.suptitle(f"{model_name} — MAE by fold and horizon")
    fig.tight_layout()
    fig.savefig(plots / f"{model_name}_mae_by_fold.png", dpi=150)
    plt.close(fig)
    print(f"Saved {model_name}_mae_by_fold.png")


def plot_model_comparison(results_dir: Path | None = None) -> None:
    """Bar chart comparing all models across horizons."""
    plots = ensure_plots_dir()
    if results_dir is None:
        results_dir = _ROOT / "experiments" / "results"

    frames = []
    for csv in results_dir.glob("*_results.csv"):
        if "latency" in csv.name:
            continue
        frames.append(pd.read_csv(csv))
    if not frames:
        print("No results to plot")
        return

    df = pd.concat(frames, ignore_index=True)
    summary = df.groupby(["model", "horizon"])["mase"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    models = summary["model"].unique()
    horizons = sorted(summary["horizon"].unique())
    x = np.arange(len(horizons))
    width = 0.8 / len(models)

    for i, model in enumerate(sorted(models)):
        model_data = summary[summary["model"] == model]
        values = [model_data[model_data["horizon"] == h]["mase"].values[0] for h in horizons]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="MASE=1 (seasonal naive)")
    ax.set_xlabel("Forecast Horizon (hours)")
    ax.set_ylabel("MASE")
    ax.set_title("Model Comparison — MASE by Horizon")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "model_comparison_mase.png", dpi=150)
    plt.close(fig)
    print("Saved model_comparison_mase.png")


def main() -> None:
    results_dir = _ROOT / "experiments" / "results"

    for csv in results_dir.glob("*_results.csv"):
        if "latency" in csv.name:
            continue
        df = pd.read_csv(csv)
        for model in df["model"].unique():
            plot_residuals_over_time(csv, model)

    plot_model_comparison()


if __name__ == "__main__":
    main()
