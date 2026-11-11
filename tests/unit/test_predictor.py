"""The serving Predictor on a real exported bundle."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from forecast.serve.bundle import read_card
from forecast.serve.predictor import InputError, Predictor


@pytest.fixture(scope="module")
def predictor(smoke_bundle):
    return Predictor.from_dir(smoke_bundle)


LAST = datetime(2025, 4, 1, 12, tzinfo=UTC)


def history(n=168):
    t = np.arange(n)
    return 1000 + 200 * np.sin(2 * np.pi * t / 24)


def test_card_is_complete(smoke_bundle):
    card = read_card(smoke_bundle)
    assert card.input_window == 168 and card.max_horizon == 24
    assert set(card.intervals.levels) == {"0.8", "0.95"}
    assert card.training["parity_max_abs_diff"] < 1e-4
    assert "mase_mean" in card.metrics["holdout"]


def test_forecast_shape_timestamps_and_interval_order(predictor):
    fc = predictor.predict(history(), LAST, horizon=24)
    assert len(fc.point) == 24 and fc.timestamps[0] == datetime(2025, 4, 1, 13, tzinfo=UTC)
    assert np.all(fc.lower <= fc.point) and np.all(fc.point <= fc.upper)
    wide = predictor.predict(history(), LAST, horizon=24, level="0.95")
    assert np.all(wide.upper - wide.lower >= fc.upper - fc.lower - 1e-9)


def test_uses_only_the_last_window(predictor):
    long = np.concatenate([np.full(500, 5.0), history()])
    a = predictor.predict(long, LAST, 6).point
    b = predictor.predict(history(), LAST, 6).point
    np.testing.assert_allclose(a, b)


def test_batch_matches_single(predictor):
    windows = np.stack([history(), history() * 1.1])
    hours = np.zeros((2, 168), dtype=np.int64)
    batch = predictor.predict_windows(windows, hours)
    single = predictor.predict(history() * 1.1, LAST, 24).point
    np.testing.assert_allclose(batch[1], single, rtol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"history": history(100)}, "at least 168"),
        ({"horizon": 25}, "horizon"),
        ({"horizon": 0}, "horizon"),
        ({"level": "0.5"}, "not calibrated"),
        ({"history": np.r_[history(167), np.nan]}, "NaN"),
    ],
)
def test_input_errors(predictor, kwargs, match):
    args = {"history": history(), "last_timestamp": LAST, "horizon": 24, **kwargs}
    with pytest.raises(InputError, match=match):
        predictor.predict(**args)
