"""Download UCI ElectricityLoadDiagrams20112014 dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "config" / "data.yaml"
_RAW_DIR = _ROOT / "data" / "raw"


def load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


def download(force: bool = False) -> Path:
    cfg = load_config()
    url = cfg["dataset"]["source_url"]
    raw_filename = cfg["dataset"]["raw_filename"]

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = _RAW_DIR / raw_filename

    if dest.exists() and not force:
        print(f"Already downloaded: {dest}")
        return dest

    zip_path = _RAW_DIR / "download.zip"
    print(f"Downloading from {url} ...")
    resp = requests.get(url, timeout=(30, 600), stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb = downloaded / 1e6
                total_mb = total / 1e6
                print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)
    print()

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        txt_files = [n for n in names if n.endswith(".txt")]
        if not txt_files:
            raise FileNotFoundError(f"No .txt file found in archive. Contents: {names}")
        extract_name = txt_files[0]
        print(f"Extracting {extract_name} -> {dest}")
        with zf.open(extract_name) as src, open(dest, "wb") as dst:
            dst.write(src.read())

    zip_path.unlink(missing_ok=True)
    print(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    download()
