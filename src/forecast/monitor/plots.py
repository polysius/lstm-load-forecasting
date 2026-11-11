"""Static figure of a planted-shift simulation (complements the Grafana screenshot)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_simulation(summary: dict[str, Any], out: Path) -> Path:
    events = summary["events"]
    mon = [e for e in events if e["kind"] == "monitor"]
    days = [e["sim_day"] for e in mon]
    shift_day = next((e["sim_day"] for e in events if e["kind"] == "shift_planted"), None)

    def series(key: str) -> list[float]:
        return [e["detail"].get(key, float("nan")) for e in mon]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax1.plot(days, series("psi"), marker="o", ms=3, label="PSI, quantile bins (input drift)")
    ax1.plot(
        days,
        series("threshold"),
        color="tab:red",
        ls="--",
        lw=1.2,
        label="threshold: q99 of the PSI's null on history",
    )
    ax1.set_ylabel("PSI")
    ax1.set_yscale("log")
    ax2.plot(days, series("mase_h1"), marker="o", ms=3, label="7-day MASE h=1")
    ax2.plot(days, series("mase_h24"), marker="o", ms=3, label="7-day MASE h=24")
    ax2.plot(
        days,
        series("mase_trigger"),
        color="tab:red",
        ls="--",
        lw=1.2,
        label="trigger: 1.25 x validated MASE h=24",
    )
    ax2.axhline(1.0, color="grey", ls=":", lw=1, label="seasonal naive")
    ax2.set_ylabel("rolling MASE (deployed model)")
    ax2.set_xlabel("simulated day")
    for ax in (ax1, ax2):
        if shift_day is not None:
            ax.axvline(shift_day, color="black", lw=1.2)
        for e in events:
            if e["kind"] == "retrain_triggered":
                ax.axvline(e["sim_day"], color="tab:orange", lw=1, ls="-.")
            if e["kind"] == "deployed" and e["sim_day"] > 0:
                ax.axvline(e["sim_day"], color="tab:green", lw=1.8)
        ax.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)
    ax1.set_title(
        f"Planted x{summary['shift_factor']} shift (black): orange = retrain triggered, "
        "green = retrained model promoted and deployed",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
