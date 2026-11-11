"""Input drift: PSI and KS between a reference and a current window of load values.

The decision statistic is a PSI over **quantile bins of the reference**. Evidently's
numeric PSI bins the combined sample into equal-width (Sturges) bins, so on spiky
load - one demo customer has p99 5 kW and max 31 kW - nearly all mass lands in the
first bin and a 25% level shift scores PSI 0.01. Quantile bins follow the bulk of
the distribution. Evidently still renders the HTML report and supplies its own PSI
and the KS p-value, which are reported alongside. The KS statistic comes from scipy
(Evidently reports only the p-value, which underflows under any real shift).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftResult:
    psi: float  # quantile-binned: the decision statistic
    psi_evidently: float  # Evidently's equal-width-bin PSI, reported for comparison
    ks_statistic: float
    ks_pvalue: float  # Evidently
    n_reference: int
    n_current: int

    def to_dict(self) -> dict:
        return asdict(self)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins of the reference distribution."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("PSI needs non-empty reference and current samples")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def evidently_scores(
    reference: np.ndarray, current: np.ndarray, html: Path | None = None
) -> tuple[float, float]:
    """(PSI, KS p-value) from Evidently; optionally saves the HTML report."""
    import pandas as pd
    from evidently import Report
    from evidently.metrics import ValueDrift

    ref = pd.DataFrame({"load": np.asarray(reference, dtype=float)}).dropna()
    cur = pd.DataFrame({"load": np.asarray(current, dtype=float)}).dropna()
    report = Report(
        [ValueDrift(column="load", method="psi"), ValueDrift(column="load", method="ks")]
    )
    snapshot = report.run(reference_data=ref, current_data=cur)
    values = {m["config"]["method"]: float(m["value"]) for m in snapshot.dict()["metrics"]}
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html))
    return values["psi"], values["ks"]


def drift_scores(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10, html: Path | None = None
) -> DriftResult:
    from scipy.stats import ks_2samp

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    try:
        psi_e, ks_p = evidently_scores(ref, cur, html)
    except ImportError:  # pragma: no cover - evidently is in the monitor extra
        log.warning("evidently not installed; reporting NumPy PSI only")
        psi_e, ks_p = float("nan"), float("nan")
    return DriftResult(
        psi=psi(ref, cur, bins),
        psi_evidently=psi_e,
        ks_statistic=float(ks_2samp(ref, cur).statistic),
        ks_pvalue=ks_p,
        n_reference=len(ref),
        n_current=len(cur),
    )
