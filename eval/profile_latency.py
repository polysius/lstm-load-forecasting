"""Profile and optimize LSTM inference latency (SPEC §8.3)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from models.lstm import LSTMModel

_ROOT = Path(__file__).resolve().parents[1]


CPU_VIABLE = {
    "hidden_size": 32,
    "num_layers": 1,
}


def load_configs() -> tuple[dict, dict]:
    with open(_ROOT / "config" / "lstm.yaml") as f:
        lstm_cfg = yaml.safe_load(f)
    with open(_ROOT / "config" / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    return lstm_cfg, data_cfg


def benchmark(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_runs: int = 1000,
    warmup: int = 100,
    label: str = "",
) -> dict[str, float]:
    """Measure p50/p95 latency over n_runs after warmup."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(x)
            latencies.append((time.perf_counter() - start) * 1000)

    latencies_arr = np.array(latencies)
    result = {
        "label": label,
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "mean_ms": float(np.mean(latencies_arr)),
    }
    return result


def profile_single(model, x, label):
    result = benchmark(model, x, label=label)
    print(f"  {label:30s}  p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms")
    return result


def main() -> None:
    lstm_cfg, data_cfg = load_configs()
    input_window = data_cfg["task"]["input_window"]
    max_horizon = max(data_cfg["task"]["forecast_horizons"])

    print("=== LSTM Inference Latency Profiling ===")
    print(f"Input window: {input_window}, Max horizon: {max_horizon}")
    print()

    # Create model (CPU-viable config matching trained model)
    model = LSTMModel(
        input_size=1,
        hidden_size=CPU_VIABLE["hidden_size"],
        num_layers=CPU_VIABLE["num_layers"],
        dropout=0.0,
        max_horizon=max_horizon,
    )
    model.eval()

    results = []

    # Single sequence
    x_single = torch.randn(1, input_window, 1)
    print("--- Single sequence (batch=1) ---")
    results.append(profile_single(model, x_single, "baseline_cpu_single"))

    # Batched
    for batch_size in [16, 64, 256]:
        x_batch = torch.randn(batch_size, input_window, 1)
        results.append(profile_single(model, x_batch, f"baseline_cpu_batch{batch_size}"))

    # Optimization 1: torch.no_grad + inference_mode
    print("\n--- With torch.inference_mode ---")
    with torch.inference_mode():
        results.append(profile_single(model, x_single, "inference_mode_single"))

    # Optimization 2: JIT trace
    print("\n--- TorchScript (JIT trace) ---")
    try:
        traced = torch.jit.trace(model, x_single)
        results.append(profile_single(traced, x_single, "jit_traced_single"))
        for batch_size in [64, 256]:
            x_batch = torch.randn(batch_size, input_window, 1)
            results.append(profile_single(traced, x_batch, f"jit_traced_batch{batch_size}"))
    except Exception as e:
        print(f"  JIT trace failed: {e}")

    # Optimization 3: ONNX export + runtime
    print("\n--- ONNX Runtime ---")
    try:
        import onnxruntime as ort

        onnx_path = _ROOT / "experiments" / "sweeps" / "lstm_model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            x_single,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        session = ort.InferenceSession(str(onnx_path))

        def onnx_predict(x_np):
            return session.run(None, {"input": x_np})

        # Warmup
        x_np = x_single.numpy()
        for _ in range(100):
            onnx_predict(x_np)

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            onnx_predict(x_np)
            latencies.append((time.perf_counter() - start) * 1000)

        arr = np.array(latencies)
        result = {
            "label": "onnx_runtime_single",
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(np.mean(arr)),
        }
        results.append(result)
        print(f"  {'onnx_runtime_single':30s}  "
              f"p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms")

        onnx_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"  ONNX export/runtime failed: {e}")

    # Save results
    import pandas as pd

    df = pd.DataFrame(results)
    out_path = _ROOT / "experiments" / "results" / "latency_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("LATENCY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':30s}  {'p50 (ms)':>10s}  {'p95 (ms)':>10s}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:30s}  {r['p50_ms']:10.2f}  {r['p95_ms']:10.2f}")


if __name__ == "__main__":
    main()
