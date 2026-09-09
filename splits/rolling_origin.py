"""Rolling-origin (expanding-window) cross-validation with purge gaps (SPEC §8.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "config" / "data.yaml"


def load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


@dataclass
class Fold:
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_size: int
    n_boundary_dropped: int = 0


def generate_folds(
    n_samples: int,
    n_folds: int = 5,
    input_window: int | None = None,
    max_horizon: int | None = None,
    min_train_size: int | None = None,
) -> list[Fold]:
    """Generate expanding-window folds with purge gaps.

    The purge gap between train and test is (input_window + max_horizon) to prevent
    any training window from overlapping a test target (SPEC §5.4).
    """
    cfg = load_config()
    if input_window is None:
        input_window = cfg["task"]["input_window"]
    if max_horizon is None:
        max_horizon = max(cfg["task"]["forecast_horizons"])

    purge_size = input_window + max_horizon
    if min_train_size is None:
        min_train_size = input_window * 4

    usable = n_samples - purge_size
    test_size = (usable - min_train_size) // n_folds

    if test_size < max_horizon:
        raise ValueError(
            f"Not enough data for {n_folds} folds. "
            f"n_samples={n_samples}, min_train={min_train_size}, purge={purge_size}"
        )

    folds = []
    for i in range(n_folds):
        train_end = min_train_size + i * test_size
        test_start = train_end + purge_size
        test_end = min(test_start + test_size, n_samples)

        if test_end <= test_start:
            break

        fold = Fold(
            fold_id=i,
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            purge_size=purge_size,
        )
        folds.append(fold)

    if len(folds) < n_folds:
        print(f"Warning: only {len(folds)} folds generated (requested {n_folds})")

    return folds


def split_data(
    df: pd.DataFrame,
    fold: Fold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by fold indices."""
    train = df.iloc[fold.train_start : fold.train_end]
    test = df.iloc[fold.test_start : fold.test_end]
    return train, test
