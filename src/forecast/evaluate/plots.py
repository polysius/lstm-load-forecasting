"""Diagnostic plots for a scored benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from forecast.evaluate.report import EvaluationReport


def plot_mae_by_fold(table, model: str, out: Path) -> None:
    sub = table[table["model"] == model]
    horizons = sorted(sub["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), squeeze=False)
    for ax, h in zip(axes[0], horizons, strict=True):
        g = sub[sub["horizon"] == h].groupby("fold")["mae"].mean()
        ax.bar(g.index, g.values, alpha=0.8)
        ax.set_title(f"h={h}")
        ax.set_xlabel("fold")
        ax.set_ylabel("MAE (z)")
    fig.suptitle(f"{model}: MAE by fold and horizon")
    fig.tight_layout()
    fig.savefig(out / f"{model}_mae_by_fold.png", dpi=130)
    plt.close(fig)


def plot_model_comparison(table, out: Path, title: str) -> None:
    summary = table.groupby(["model", "horizon"])["mase"].mean().reset_index()
    models = sorted(summary["model"].unique())
    horizons = sorted(summary["horizon"].unique())
    x = np.arange(len(horizons))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, model in enumerate(models):
        vals = [
            summary[(summary.model == model) & (summary.horizon == h)]["mase"].mean()
            for h in horizons
        ]
        ax.bar(x + i * width, vals, width, label=model, alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="MASE = 1 (seasonal naive)")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison_mase.png", dpi=130)
    plt.close(fig)


def plot_benchmark(report: EvaluationReport, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    combined = report.combined
    for model in sorted(combined["model"].unique()):
        plot_mae_by_fold(combined, model, out)
    plot_model_comparison(combined, out, f"{report.benchmark}: MASE by horizon")
