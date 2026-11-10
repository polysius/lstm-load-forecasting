"""Split-conformal prediction intervals, one quantile per step ahead.

Absolute residuals on a calibration span the model never trained on give, for
each step ``k`` and level ``1 - alpha``, the half-width
``q_k = the ceil((n + 1)(1 - alpha))-th smallest |residual_k|``.
The finite-sample guarantee assumes exchangeable residuals; hourly load is not
exchangeable, so coverage is *monitored* in production rather than assumed.
"""

from __future__ import annotations

import math

import numpy as np


def level_key(level: float) -> str:
    return f"{level:g}"


def conformal_halfwidths(abs_residuals: np.ndarray, levels) -> dict[str, list[float]]:
    """``(n, H)`` absolute residuals -> ``{level: [q_1 .. q_H]}``."""
    res = np.asarray(abs_residuals, dtype=np.float64)
    if res.ndim != 2 or len(res) == 0:
        raise ValueError(f"need a non-empty (n, H) residual matrix, got {res.shape}")
    n = len(res)
    ordered = np.sort(res, axis=0)
    out = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), got {level}")
        rank = min(math.ceil((n + 1) * level), n)  # clamp: tiny n cannot reach the guarantee
        out[level_key(level)] = ordered[rank - 1].tolist()
    return out
