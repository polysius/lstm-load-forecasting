"""Render experiment results CSVs into README tables (SPEC §12.4)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _ROOT / "experiments" / "results"
_README = _ROOT / "README.md"

PLACEHOLDER = "<!-- populated by scripts/render_results.py -->"


def load_results() -> pd.DataFrame | None:
    csvs = sorted(_RESULTS_DIR.glob("*.csv"))
    if not csvs:
        return None
    frames = [pd.read_csv(c) for c in csvs]
    return pd.concat(frames, ignore_index=True)


def format_results_table(df: pd.DataFrame) -> str:
    """Format a model x horizon results table."""
    metrics = ["mae", "rmse", "mase"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return "*No metrics found in results.*"

    lines = []
    for horizon in sorted(df["horizon"].unique()):
        lines.append(f"\n### h = {horizon}\n")
        subset = df[df["horizon"] == horizon]
        header = "| Model | " + " | ".join(m.upper() for m in available) + " |"
        sep = "|---|" + "|".join(["---"] * len(available)) + "|"
        lines.append(header)
        lines.append(sep)
        for _, row in subset.iterrows():
            vals = " | ".join(f"{row.get(m, 'N/A'):.4f}" if pd.notna(row.get(m)) else "N/A" for m in available)
            lines.append(f"| {row['model']} | {vals} |")

    return "\n".join(lines)


def render() -> None:
    df = load_results()
    if df is None:
        print("No results CSVs found. Skipping render.")
        return

    table = format_results_table(df)
    readme_text = _README.read_text(encoding="utf-8")

    # Replace the first placeholder with the results table
    if PLACEHOLDER in readme_text:
        readme_text = readme_text.replace(PLACEHOLDER, table, 1)
        _README.write_text(readme_text, encoding="utf-8")
        print(f"Updated {_README}")
    else:
        print("No placeholder found in README.md")


if __name__ == "__main__":
    render()
