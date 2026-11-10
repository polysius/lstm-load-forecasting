"""PyTorch -> ONNX export with a numerical parity gate."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

PARITY_ATOL = 1e-4  # in standardised units; observed differences are ~1e-7


def export_onnx(net: torch.nn.Module, path: Path, input_window: int, n_features: int) -> str:
    """Export with the dynamo exporter (dynamic batch), falling back to TorchScript."""
    net = net.eval()
    # torch.export specialises size-1 dims, so trace with batch 2 to keep batch dynamic.
    dummy = torch.zeros(2, input_window, n_features)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = torch.export.Dim("batch", min=1, max=65536)
            torch.onnx.export(
                net,
                (dummy,),
                str(path),
                input_names=["x"],
                output_names=["y"],
                dynamic_shapes={"x": {0: batch}},
                dynamo=True,
                external_data=False,  # one self-contained file; the model is tiny
                verbose=False,
            )
        return "dynamo"
    except Exception as exc:  # pragma: no cover - depends on the torch build
        log.warning("dynamo export failed (%s); falling back to TorchScript exporter", exc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                net,
                (dummy,),
                str(path),
                input_names=["x"],
                output_names=["y"],
                dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
                dynamo=False,
                opset_version=17,
            )
        return "torchscript"


def parity_max_abs_diff(
    net: torch.nn.Module, path: Path, input_window: int, n_features: int, n: int = 64
) -> float:
    import onnxruntime as ort

    x = torch.from_numpy(
        np.random.default_rng(0).standard_normal((n, input_window, n_features)).astype("float32")
    )
    with torch.no_grad():
        expected = net.eval()(x).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    return float(np.max(np.abs(got - expected)))


def export_with_parity(
    net: torch.nn.Module, path: Path, input_window: int, n_features: int
) -> dict[str, float | str]:
    exporter = export_onnx(net, path, input_window, n_features)
    diff = parity_max_abs_diff(net, path, input_window, n_features)
    if diff > PARITY_ATOL:
        raise RuntimeError(f"ONNX parity check failed: max |diff| = {diff:.2e} > {PARITY_ATOL}")
    log.info("exported %s via %s exporter (parity max |diff| %.1e)", path.name, exporter, diff)
    return {"exporter": exporter, "parity_max_abs_diff": diff}
