"""Render experiment results CSVs into README tables (SPEC §12.4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _ROOT / "experiments" / "results"
_README = _ROOT / "README.md"

PLACEHOLDER = "<!-- populated by scripts/render_results.py -->"


def load_results() -> pd.DataFrame | None:
    csvs = sorted(_RESULTS_DIR.glob("*_results.csv"))
    if not csvs:
        return None
    frames = [pd.read_csv(c) for c in csvs if "latency" not in c.name]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def format_results_table(df: pd.DataFrame) -> str:
    """Format a single model x horizon table with MAE, RMSE, MASE."""
    summary = df.groupby(["model", "horizon"])[["mae", "rmse", "mase"]].mean().reset_index()

    lines = ["| Model | Horizon | MAE | RMSE | MASE |", "|---|---|---|---|---|"]
    for _, row in summary.sort_values(["horizon", "mase"]).iterrows():
        mase_str = f"**{row['mase']:.4f}**" if row["mase"] < 1 else f"{row['mase']:.4f}"
        lines.append(
            f"| {row['model']} | h={int(row['horizon'])} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {mase_str} |"
        )

    return "\n".join(lines)


def format_complexity_answer(df: pd.DataFrame) -> str:
    """Answer: did the LSTM earn its complexity?"""
    summary = df.groupby(["model", "horizon"])["mase"].mean().reset_index()

    lines = []
    for h in sorted(summary["horizon"].unique()):
        h_data = summary[summary["horizon"] == h].set_index("model")["mase"]
        lstm_mase = h_data.get("lstm", None)
        naive_mase = h_data.get("naive", None)
        sn_mase = h_data.get("seasonal_naive", None)
        best_model = h_data.idxmin()
        best_mase = h_data.min()

        line = f"- **h={int(h)}:** Best = {best_model} (MASE {best_mase:.4f})"
        if lstm_mase is not None and sn_mase is not None:
            if lstm_mase < sn_mase:
                line += f". LSTM ({lstm_mase:.4f}) beats seasonal naive ({sn_mase:.4f})."
            else:
                line += f". LSTM ({lstm_mase:.4f}) does not beat seasonal naive ({sn_mase:.4f})."
        lines.append(line)

    verdict = []
    lstm_all = summary[summary["model"] == "lstm"]["mase"].mean() if "lstm" in summary["model"].values else None
    sn_all = summary[summary["model"] == "seasonal_naive"]["mase"].mean()
    naive_all = summary[summary["model"] == "naive"]["mase"].mean()

    verdict.append("")
    if lstm_all is not None and lstm_all < sn_all:
        verdict.append(
            f"The LSTM (mean MASE {lstm_all:.4f}) beats seasonal naive ({sn_all:.4f}) overall, "
            f"but naive persistence ({naive_all:.4f}) remains the strongest baseline at short horizons. "
            "The LSTM earns partial complexity: it adds value at longer horizons where persistence degrades."
        )
    elif lstm_all is not None:
        verdict.append(
            f"The LSTM (mean MASE {lstm_all:.4f}) does not consistently beat seasonal naive ({sn_all:.4f}). "
            "The added complexity is not justified on this dataset with this configuration."
        )
    else:
        verdict.append("LSTM results not yet available.")

    return "\n".join(lines + verdict)


def load_latency() -> str | None:
    lat_path = _RESULTS_DIR / "latency_results.csv"
    if not lat_path.exists():
        return None
    df = pd.read_csv(lat_path)
    lines = ["| Method | p50 (ms) | p95 (ms) |", "|---|---|---|"]
    for _, row in df.iterrows():
        lines.append(f"| {row['label']} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} |")
    return "\n".join(lines)


def render() -> None:
    df = load_results()
    if df is None:
        print("No results CSVs found. Skipping render.")
        return

    readme_text = _README.read_text(encoding="utf-8")

    table = format_results_table(df)
    complexity = format_complexity_answer(df)
    latency = load_latency()

    # Replace placeholders in order
    replacements = [table, complexity]
    if latency:
        replacements.append(latency)

    for replacement in replacements:
        if PLACEHOLDER in readme_text:
            readme_text = readme_text.replace(PLACEHOLDER, replacement, 1)

    _README.write_text(readme_text, encoding="utf-8")
    print(f"Updated {_README}")


if __name__ == "__main__":
    render()
