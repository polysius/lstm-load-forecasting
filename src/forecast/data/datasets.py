"""Build and load versioned datasets for any configured source."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from forecast.config import DataConfig
from forecast.data.validate import validate
from forecast.data.versioning import Snapshot, latest_snapshot, load_snapshot, write_snapshot

log = logging.getLogger(__name__)


class DatasetNotFound(FileNotFoundError):
    pass


def build_dataset(cfg: DataConfig, data_dir: Path, *, force_download: bool = False) -> Snapshot:
    """Fetch/preprocess per the config, validate the contract, write a snapshot."""
    if cfg.source == "uci":
        from forecast.data import uci

        raw = uci.download(cfg, data_dir, force=force_download)
        df = uci.preprocess(cfg, raw)
        source = cfg.uci.source_url if cfg.uci else ""
        extra = {"raw_file": raw.name, "raw_version": raw.parent.name}
    elif cfg.source == "eia":
        from forecast.data import eia

        df, extra = eia.fetch_history(cfg)
        source = eia.public_source(cfg)
    else:  # pragma: no cover - guarded by the config schema
        raise ValueError(cfg.source)

    summary = validate(df, freq=cfg.preprocessing.resample_freq)
    snap = write_snapshot(
        df,
        dataset=cfg.name,
        data_dir=data_dir,
        source=source,
        extra={**extra, "validation": summary, "units": cfg.units},
    )
    log.info(
        "dataset %s version %s sha256=%s rows=%d",
        snap.dataset,
        snap.version,
        snap.sha256[:12],
        snap.rows,
    )
    return snap


def load_series(
    cfg: DataConfig, data_dir: Path, *, version: str | None = None
) -> tuple[pd.Series, Snapshot]:
    """The target series of the latest (or a pinned) snapshot, hash-verified."""
    if version is None:
        snap = latest_snapshot(cfg.name, data_dir)
    else:
        from forecast.data.versioning import list_snapshots

        snap = next((s for s in list_snapshots(cfg.name, data_dir) if s.version == version), None)
    if snap is None:
        raise DatasetNotFound(
            f"no snapshot for dataset {cfg.name!r} under {data_dir}; run `forecast data {cfg.name}`"
        )
    df = load_snapshot(snap)
    return df[cfg.target_col], snap
