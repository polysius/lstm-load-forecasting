"""Prediction logging sinks. One JSON record per served forecast, inputs included.

``PREDICTION_LOG_URI`` selects the sink:
  * unset / ``none``        -> disabled
  * ``file:///path`` or path -> JSON lines under ``<path>/date=YYYY-MM-DD/``
  * ``s3://bucket/prefix``   -> one object per request under ``prefix/date=YYYY-MM-DD/``

Records are partitioned by the forecast *origin* date, which is what the
monitoring job joins against actuals.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class PredictionLogger(Protocol):
    def log(self, record: dict[str, Any]) -> None: ...


def _partition(record: dict[str, Any]) -> str:
    return f"date={str(record['origin'])[:10]}"


class NullLogger:
    def log(self, record: dict[str, Any]) -> None:
        return None


class JsonlLogger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def log(self, record: dict[str, Any]) -> None:
        directory = self.root / _partition(record)
        directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with self._lock, open(directory / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write(line)


class S3Logger:
    def __init__(self, bucket: str, prefix: str, client=None):
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def log(self, record: dict[str, Any]) -> None:
        key = f"{self.prefix}/{_partition(record)}/{record['request_id']}.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key.lstrip("/"),
            Body=json.dumps(record, default=str).encode(),
            ContentType="application/json",
        )


def from_uri(uri: str | None) -> PredictionLogger:
    if not uri or uri.lower() == "none":
        return NullLogger()
    if uri.startswith("s3://"):
        bucket, _, prefix = uri[5:].partition("/")
        return S3Logger(bucket, prefix or "predictions")
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return JsonlLogger(Path(uri))


def read_jsonl(root: Path) -> list[dict[str, Any]]:
    """All records under a JsonlLogger root (used by the local monitoring job)."""
    records = []
    for path in sorted(Path(root).glob("date=*/predictions.jsonl")):
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records
