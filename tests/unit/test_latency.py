"""The latency benchmark emits the schema of the published experiments/results/latency.csv."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.slow  # the ONNX export alone takes about 15 s

from forecast.evaluate.latency import benchmark, to_markdown  # noqa: E402

PUBLISHED_COLUMNS = [
    "method",
    "threads",
    "batch",
    "p50_ms",
    "p95_ms",
    "mean_ms",
    "per_sequence_p50_ms",
    "cpu",
    "cores",
    "torch",
    "onnxruntime",
]
METHOD_ORDER = [
    "pytorch_eager_no_grad",
    "pytorch_inference_mode",
    "onnxruntime",
    "serving_predictor_end_to_end",
]


@pytest.fixture(scope="module")
def result():
    import torch

    default_threads = torch.get_num_threads()
    df = benchmark(hidden_size=4, input_window=24, max_horizon=6, runs=3, warmup=1, batches=(1, 2))
    return df, default_threads, torch.get_num_threads()


def test_columns_and_methods_match_the_published_csv(result):
    df, _, _ = result
    assert list(df.columns) == PUBLISHED_COLUMNS
    assert set(df["method"]) == set(METHOD_ORDER)


def test_each_runtime_is_timed_single_threaded_and_at_its_default(result):
    df, default_threads, _ = result
    threads = df.groupby("method")["threads"].apply(set).to_dict()
    assert threads["pytorch_eager_no_grad"] == {1, default_threads}
    assert threads["pytorch_inference_mode"] == {1, default_threads}
    assert threads["onnxruntime"] == {1, os.cpu_count()}
    assert threads["serving_predictor_end_to_end"] == {1}
    serving = df[df["method"] == "serving_predictor_end_to_end"]
    assert serving["batch"].tolist() == [1]


def test_rows_are_ordered_by_batch_then_method_then_threads(result):
    df, _, _ = result
    keys = list(zip(df["batch"], df["method"].map(METHOD_ORDER.index), df["threads"], strict=True))
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_torch_thread_count_is_restored(result):
    _, before, after = result
    assert after == before


def test_markdown_has_one_line_per_row(result):
    df, _, _ = result
    assert len(to_markdown(df).splitlines()) == len(df) + 2
