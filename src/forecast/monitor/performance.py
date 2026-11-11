"""Performance drift: join logged predictions with actuals, score a rolling window."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def predictions_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (request, step) from prediction-log records."""
    rows = []
    for rec in records:
        for f in rec["forecasts"]:
            rows.append(
                {
                    "request_id": rec["request_id"],
                    "model_version": str(rec.get("model_version", "")),
                    "origin": rec["origin"],
                    "step": int(f["step"]),
                    "target_time": f["timestamp"],
                    "point": float(f["point"]),
                    "lower": float(f["lower"]),
                    "upper": float(f["upper"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "request_id",
                "model_version",
                "origin",
                "step",
                "target_time",
                "point",
                "lower",
                "upper",
            ]
        )
    df = pd.DataFrame(rows)
    df["origin"] = pd.to_datetime(df["origin"], utc=True, format="ISO8601")
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True, format="ISO8601")
    return df


def join_actuals(preds: pd.DataFrame, actuals: pd.Series) -> pd.DataFrame:
    """Attach the realised value to every predicted step (NaN until it is observed)."""
    act = actuals.copy()
    act.index = pd.DatetimeIndex(act.index).tz_convert("UTC")
    out = preds.copy()
    out["actual"] = act.reindex(pd.DatetimeIndex(out["target_time"])).to_numpy()
    out["abs_error"] = (out["actual"] - out["point"]).abs()
    out["covered"] = (out["actual"] >= out["lower"]) & (out["actual"] <= out["upper"])
    return out


def rolling_metrics(
    joined: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scale: float,
    horizons: tuple[int, ...] = (1, 24),
) -> dict[str, float]:
    """MASE (steps 1..h pooled, as in the backtest), MAE, coverage; origins in [start, end)."""
    window = joined[(joined["origin"] >= start) & (joined["origin"] < end)]
    scored = window.dropna(subset=["actual"])
    out: dict[str, float] = {
        "n_requests": float(window["request_id"].nunique()),
        "n_scored_points": float(len(scored)),
    }
    for h in horizons:
        sub = scored[scored["step"] <= h]
        mae = float(sub["abs_error"].mean()) if len(sub) else float("nan")
        out[f"mae_h{h}"] = mae
        out[f"mase_h{h}"] = mae / scale if np.isfinite(mae) else float("nan")
    out["coverage"] = float(scored["covered"].mean()) if len(scored) else float("nan")
    return out
