"""UCI ElectricityLoadDiagrams20112014: download and preprocess one client to hourly.

The preprocessing is a faithful port of the original ``data/preprocess.py`` so the
benchmark reproduces the published results bit-for-bit on the same raw file.
"""

from __future__ import annotations

import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from forecast.config import DataConfig
from forecast.data.cleaning import clean_hourly

log = logging.getLogger(__name__)


def raw_root(data_dir: Path) -> Path:
    return data_dir / "raw" / "uci"


def find_raw(cfg: DataConfig, data_dir: Path) -> Path | None:
    """Most recent dated download of the raw file, if any."""
    assert cfg.uci is not None
    hits = sorted(raw_root(data_dir).glob(f"*/{cfg.uci.raw_filename}"))
    return hits[-1] if hits else None


def download(cfg: DataConfig, data_dir: Path, *, force: bool = False) -> Path:
    """Download the archive into ``data/raw/uci/<YYYY-MM-DD>/`` (dated prefix)."""
    import requests

    assert cfg.uci is not None
    existing = find_raw(cfg, data_dir)
    if existing is not None and not force:
        log.info("raw file already present: %s", existing)
        return existing

    dest_dir = raw_root(data_dir) / datetime.now(UTC).strftime("%Y-%m-%d")
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "download.zip"
    log.info("downloading %s", cfg.uci.source_url)
    with requests.get(cfg.uci.source_url, timeout=(30, 600), stream=True) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    dest = dest_dir / cfg.uci.raw_filename
    with zipfile.ZipFile(zip_path) as zf:
        txt = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt:
            raise FileNotFoundError(f"no .txt in archive: {zf.namelist()}")
        with zf.open(txt[0]) as src, open(dest, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    zip_path.unlink(missing_ok=True)
    log.info("saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def load_raw_client(path: Path, client: str) -> pd.Series:
    """Read one client column. Semicolon-separated, comma decimal, 15-minute kW."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(";")
    names = [h.strip('"') for h in header]
    if client not in names:
        raise ValueError(f"client {client!r} not in dataset; first columns: {names[1:11]}")
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
        usecols=[0, names.index(client)],
    )
    series = df[client].astype("float64")
    series.index.name = "timestamp"
    return series


def preprocess(cfg: DataConfig, raw_path: Path) -> pd.DataFrame:
    """Raw 15-minute readings for one client -> clean, gap-explicit hourly frame."""
    assert cfg.uci is not None
    pp = cfg.preprocessing
    series = load_raw_client(raw_path, cfg.uci.target_client)
    log.info("raw: %d rows, %s -> %s", len(series), series.index[0], series.index[-1])

    # The raw timestamps are local Portuguese time with DST ambiguities. The
    # original pipeline localises as UTC (keeping the numbers) and converts; keep it.
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC").tz_convert(pp.timezone)
    else:
        series.index = series.index.tz_convert(pp.timezone)

    hourly = clean_hourly(series, pp, name=cfg.target_col)
    return hourly.to_frame()
