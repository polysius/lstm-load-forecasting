"""CPU inference latency: PyTorch eager vs ONNX Runtime, plus the full serving path.

Latency does not depend on the weights, so the benchmark uses the production
architecture (from a bundle's model card when given, else the LSTM config).
"""

from __future__ import annotations

import os
import platform
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _time(fn: Callable[[], object], runs: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        out[i] = (time.perf_counter() - t0) * 1000
    return out


def _row(label: str, batch: int, ms: np.ndarray) -> dict:
    return {
        "method": label,
        "batch": batch,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "per_sequence_p50_ms": float(np.percentile(ms, 50)) / batch,
    }


def benchmark(
    *,
    hidden_size: int = 32,
    num_layers: int = 1,
    input_window: int = 168,
    max_horizon: int = 24,
    runs: int = 1000,
    warmup: int = 100,
    batches: tuple[int, ...] = (1, 64),
    torch_threads: int | None = None,
) -> pd.DataFrame:
    import onnxruntime as ort
    import torch

    from forecast.models.lstm import LSTMNet
    from forecast.serve.bundle import IntervalSpec, ModelCard, ScalerParams, write_card
    from forecast.serve.predictor import Predictor
    from forecast.train.export import export_with_parity

    if torch_threads:
        torch.set_num_threads(torch_threads)
    torch.manual_seed(0)
    net = LSTMNet(1, hidden_size, num_layers, 0.0, max_horizon).eval()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export_with_parity(net, onnx_path, input_window, 1)
        sessions = {}
        for threads in (1, 0):  # 0 = ORT default (all cores)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessions[threads] = ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        for batch in batches:
            x = torch.randn(batch, input_window, 1)
            xn = x.numpy()

            def eager(x=x):
                with torch.no_grad():
                    net(x)

            def inference_mode(x=x):
                with torch.inference_mode():
                    net(x)

            rows.append(_row("pytorch_eager_no_grad", batch, _time(eager, runs, warmup)))
            rows.append(_row("pytorch_inference_mode", batch, _time(inference_mode, runs, warmup)))
            for threads, sess in sessions.items():
                label = "onnxruntime_1_thread" if threads == 1 else "onnxruntime_default_threads"
                rows.append(
                    _row(
                        label,
                        batch,
                        _time(lambda s=sess, x=xn: s.run(None, {"x": x}), runs, warmup),
                    )
                )

        card = ModelCard(
            model_name="latency",
            created_at=datetime.now(UTC).isoformat(),
            architecture={"type": "lstm"},
            input_window=input_window,
            max_horizon=max_horizon,
            scaler=ScalerParams(mean=0.0, scale=1.0),
            intervals=IntervalSpec(levels={"0.8": [1.0] * max_horizon}, default_level="0.8"),
        )
        write_card(card, Path(tmp))
        predictor = Predictor.from_dir(tmp, threads=1)
        history = np.random.default_rng(0).normal(1000, 50, input_window)
        last = datetime(2026, 1, 1, tzinfo=UTC)
        rows.append(
            _row(
                "serving_predictor_end_to_end",
                1,
                _time(lambda: predictor.predict(history, last, max_horizon), runs, warmup),
            )
        )

    df = pd.DataFrame(rows)
    df["cpu"] = platform.processor() or platform.machine()
    df["cores"] = os.cpu_count()
    df["torch"] = torch.__version__
    df["onnxruntime"] = ort.__version__
    return df


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| Method | Batch | p50 (ms) | p95 (ms) | p50 per sequence (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.method} | {r.batch} | {r.p50_ms:.3f} | {r.p95_ms:.3f} | "
            f"{r.per_sequence_p50_ms:.4f} |"
        )
    return "\n".join(lines)
