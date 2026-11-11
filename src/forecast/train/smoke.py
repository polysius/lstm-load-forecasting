"""A tiny, quickly-trained bundle on synthetic data for tests, CI and container builds.

It exercises the real export/calibration code path; its forecasts are meaningless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import ProductionConfig
from forecast.serve.bundle import write_card


def synthetic_load(n: int = 2400, seed: int = 0, level: float = 1000.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ar = np.zeros(n)
    eps = rng.normal(0, 0.02 * level, n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + eps[i]
    y = level * (1 + 0.2 * np.sin(2 * np.pi * t / 24) + 0.05 * np.sin(2 * np.pi * t / 168)) + ar
    idx = pd.date_range("2025-01-06", periods=n, freq="h", tz="UTC", name="timestamp")
    return pd.Series(y, index=idx, name="load")


def build_smoke_bundle(out_dir: Path, *, epochs: int = 3, hidden: int = 8, seed: int = 0) -> Path:
    from forecast.train.production import train_production

    cfg = ProductionConfig(
        model_name="smoke-test",
        data="eia",  # only task settings (window/horizons) are read; no network access
        lstm={
            "hidden_size": hidden,
            "max_epochs": epochs,
            "patience": epochs,
            "batch_size": 128,
            "learning_rate": 0.005,
        },
        holdout_hours=168,
        calibration_hours=168,
        seed=seed,
    )
    result = train_production(cfg, series=synthetic_load(seed=seed), bundle_dir=out_dir)
    card = result.card
    card.series = {"dataset": "synthetic", "note": "smoke-test bundle; forecasts are meaningless"}
    write_card(card, out_dir)
    return out_dir
