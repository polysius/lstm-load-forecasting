"""Rolling-origin (expanding-window) folds with purge gaps. Chronological only."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_size: int
    n_boundary_dropped: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def generate_folds(
    n_samples: int,
    n_folds: int = 5,
    input_window: int = 168,
    max_horizon: int = 24,
    min_train_size: int | None = None,
) -> list[Fold]:
    """Expanding-window folds separated by a purge gap of ``input_window + max_horizon``.

    The gap guarantees no training window (inputs or targets) overlaps any test
    target. Test windows draw their inputs from the test span only.
    """
    purge_size = input_window + max_horizon
    if min_train_size is None:
        min_train_size = input_window * 12  # ~12 weeks for seasonal estimation

    usable = n_samples - purge_size
    test_size = (usable - min_train_size) // n_folds
    if test_size < max_horizon:
        raise ValueError(
            f"Not enough data for {n_folds} folds: n_samples={n_samples}, "
            f"min_train={min_train_size}, purge={purge_size}"
        )

    folds = []
    for i in range(n_folds):
        train_end = min_train_size + i * test_size
        test_start = train_end + purge_size
        test_end = min(test_start + test_size, n_samples)
        if test_end <= test_start:
            break
        folds.append(
            Fold(
                fold_id=i,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_size=purge_size,
                n_boundary_dropped=min(input_window, train_end),
            )
        )
    if len(folds) < n_folds:
        log.warning("only %d folds generated (requested %d)", len(folds), n_folds)
    return folds


def split_data(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[fold.train_start : fold.train_end], df.iloc[fold.test_start : fold.test_end]
