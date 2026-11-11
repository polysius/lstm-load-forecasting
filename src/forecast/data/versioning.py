"""Data versioning with dated prefixes and a content hash.

Every processed dataset is written once to
``data/processed/<dataset>/<YYYY-MM-DDTHHMMSSZ>/{series.parquet,manifest.json}``
and never modified. The manifest carries a SHA-256 of the *canonical content*
(UTC nanosecond timestamps + float64 values), not of the parquet bytes, so the
hash is stable across pandas/pyarrow versions. Training runs log the prefix and
the hash, which pins exactly which data produced a model. The same layout is
mirrored to S3 by the retraining job.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAMP_FORMAT = "%Y-%m-%dT%H%M%SZ"


class DataIntegrityError(RuntimeError):
    """A snapshot's content no longer matches the hash in its manifest."""


@dataclass(frozen=True)
class Snapshot:
    dataset: str
    version: str  # the dated prefix, e.g. 2026-11-10T153000Z
    sha256: str
    rows: int
    start: str
    end: str
    path: str
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return Path(self.path).parent


def content_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        utc = idx.tz_convert("UTC") if idx.tz is not None else idx
        h.update(np.ascontiguousarray(utc.as_unit("ns").asi8).tobytes())
        h.update(str(idx.tz).encode())
    else:
        h.update(pd.util.hash_pandas_object(idx, index=False).to_numpy().tobytes())
    for col in df.columns:
        h.update(str(col).encode())
        values = df[col].to_numpy(dtype="float64", na_value=np.nan)
        values = np.where(np.isnan(values), np.nan, values)  # canonical NaN payload
        h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


def processed_root(data_dir: Path, dataset: str) -> Path:
    return data_dir / "processed" / dataset


def write_snapshot(
    df: pd.DataFrame,
    *,
    dataset: str,
    data_dir: Path,
    source: str = "",
    extra: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Snapshot:
    """Write an immutable snapshot; reuse the latest one if the content is unchanged."""
    digest = content_hash(df)
    latest = latest_snapshot(dataset, data_dir)
    if latest is not None and latest.sha256 == digest:
        return latest

    version = (now or datetime.now(UTC)).strftime(STAMP_FORMAT)
    out_dir = processed_root(data_dir, dataset) / version
    if out_dir.exists():
        raise FileExistsError(f"snapshot {out_dir} already exists with different content")
    out_dir.mkdir(parents=True)
    path = out_dir / "series.parquet"
    df.to_parquet(path)
    snap = Snapshot(
        dataset=dataset,
        version=version,
        sha256=digest,
        rows=len(df),
        start=df.index[0].isoformat(),
        end=df.index[-1].isoformat(),
        path=str(path),
        source=source,
        extra=extra or {},
    )
    manifest = {k: v for k, v in asdict(snap).items() if k != "path"}
    manifest["created_at"] = datetime.now(UTC).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return snap


def _from_manifest(manifest_path: Path) -> Snapshot:
    meta = json.loads(manifest_path.read_text())
    meta.pop("created_at", None)
    return Snapshot(path=str(manifest_path.parent / "series.parquet"), **meta)


def list_snapshots(dataset: str, data_dir: Path) -> list[Snapshot]:
    root = processed_root(data_dir, dataset)
    return [_from_manifest(p) for p in sorted(root.glob("*/manifest.json"))]


def latest_snapshot(dataset: str, data_dir: Path) -> Snapshot | None:
    snaps = list_snapshots(dataset, data_dir)
    return snaps[-1] if snaps else None


def load_snapshot(snap: Snapshot, *, verify: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(snap.path)
    if verify and content_hash(df) != snap.sha256:
        raise DataIntegrityError(f"{snap.path} does not match manifest hash {snap.sha256[:12]}")
    return df
