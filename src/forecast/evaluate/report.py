"""Score saved forecasts, check reproduction against reference results, write tables/plots."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import BenchmarkConfig, experiments_dir, project_root
from forecast.evaluate.backtest import BacktestResult, score_forecasts
from forecast.train.benchmark import forecasts_path, jobs

log = logging.getLogger(__name__)

RESULT_COLUMNS = ["mae", "rmse", "mape", "mase", "model", "horizon", "fold"]


@dataclass
class EvaluationReport:
    benchmark: str
    run_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    reproduction: pd.DataFrame | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def reproduced(self) -> bool:
        """True when every compared row passed or is a documented known difference."""
        if self.reproduction is None:
            return True
        return bool(self.reproduction["status"].isin(["PASS", "KNOWN"]).all())

    @property
    def combined(self) -> pd.DataFrame:
        return pd.concat(self.run_tables.values(), ignore_index=True)


def score_run(cfg: BenchmarkConfig, run_name: str, missing: list[str]) -> pd.DataFrame | None:
    frames = []
    for job in jobs(cfg, [run_name]):
        path = forecasts_path(cfg.name, job)
        if not path.exists():
            missing.append(f"{run_name}/{job.key}")
            continue
        table = score_forecasts(BacktestResult.load(path))
        table["model"] = job.spec.result_label
        if job.spec.seeds:
            table["seed"] = job.seed
        frames.append(table)
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    cols = RESULT_COLUMNS + (["seed"] if "seed" in table else [])
    return table[cols]


def compare_to_reference(
    reproduced: pd.DataFrame,
    reference: pd.DataFrame,
    tolerances: dict[str, float],
    known_differences: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Mean MASE per (model, horizon), reproduced vs reference, with a relative tolerance.

    Only (model, horizon) pairs present in the reference are compared. A pair outside
    tolerance is FAIL, or KNOWN when ``known_differences`` documents why the published
    number cannot come from the committed code; a pair within tolerance is PASS either way.
    """
    known = known_differences or {}
    keys = ["model", "horizon"]
    ref_means = reference.groupby(keys)["mase"].mean()
    joined = pd.DataFrame(
        {
            "reference": ref_means,
            "reproduced": reproduced.groupby(keys)["mase"].mean().reindex(ref_means.index),
        }
    )
    joined["abs_diff"] = (joined["reproduced"] - joined["reference"]).abs()
    joined["rel_diff"] = joined["abs_diff"] / joined["reference"].abs()
    models = [model for model, _ in joined.index]
    joined["tolerance"] = [tolerances.get(m, 0.0) for m in models]
    within = (joined["rel_diff"] <= joined["tolerance"]).to_numpy()
    joined["status"] = [
        "MISSING" if np.isnan(rep) else "PASS" if ok else "KNOWN" if m in known else "FAIL"
        for m, ok, rep in zip(models, within, joined["reproduced"], strict=True)
    ]
    joined["note"] = [
        known.get(m, "") if s == "KNOWN" else ""
        for m, s in zip(models, joined["status"], strict=True)
    ]

    # Largest per-fold deviation, to show whether agreement is row-level or only on average.
    row_keys = keys + ["fold"] + (["seed"] if "seed" in reference and "seed" in reproduced else [])
    rows = reference.merge(reproduced, on=row_keys, suffixes=("_ref", "_rep"))
    rows["row_diff"] = (rows["mase_rep"] - rows["mase_ref"]).abs()
    joined["max_fold_diff"] = rows.groupby(keys)["row_diff"].max()
    return joined.reset_index()


def evaluate_benchmark(cfg: BenchmarkConfig, *, write: bool = True) -> EvaluationReport:
    report = EvaluationReport(benchmark=cfg.name)
    comparisons = []
    for run in cfg.runs:
        table = score_run(cfg, run.name, report.missing)
        if table is None:
            continue
        report.run_tables[run.name] = table
        for ref in run.references:
            reference = pd.read_csv(project_root() / ref.path)
            cmp = compare_to_reference(table, reference, cfg.tolerances, ref.known_differences)
            cmp.insert(0, "reference_file", ref.label or Path(ref.path).name)
            cmp.insert(0, "run", run.name)
            comparisons.append(cmp)
    if comparisons:
        report.reproduction = pd.concat(comparisons, ignore_index=True)

    if write and report.run_tables:
        out = experiments_dir() / "results" / cfg.name
        out.mkdir(parents=True, exist_ok=True)
        for name, table in report.run_tables.items():
            table.to_csv(out / f"{name}.csv", index=False)
        (out / "SUMMARY.md").write_text(render_summary(cfg, report), encoding="utf-8")
        try:
            from forecast.evaluate.plots import plot_benchmark

            plot_benchmark(report, experiments_dir() / "plots" / cfg.name)
        except Exception:  # plots are a convenience; never fail evaluation on them
            log.exception("plotting failed")
    return report


# --------------------------------------------------------------------------- markdown


def _fmt(x: float, digits: int = 4) -> str:
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def results_table(table: pd.DataFrame) -> str:
    """Model x horizon: mean MASE (+- std over folds, or over seeds when seeded)."""
    lines = ["| Model | Horizon | MASE | MAE (z) | RMSE (z) | n |", "|---|---|---|---|---|---|"]
    seeded = "seed" in table and table["seed"].notna().any()
    for (model, h), g in sorted(
        table.groupby(["model", "horizon"]), key=lambda kv: (kv[0][1], kv[1]["mase"].mean())
    ):
        if seeded and g["seed"].notna().all():
            per_seed = g.groupby("seed")["mase"].mean()
            spread, n = per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0, f"{len(per_seed)} seeds"
        else:
            spread, n = g["mase"].std(ddof=1) if len(g) > 1 else 0.0, f"{len(g)} folds"
        mase = g["mase"].mean()
        mase_s = f"{'**' if mase < 1 else ''}{mase:.4f}{'**' if mase < 1 else ''} ± {spread:.4f}"
        lines.append(
            f"| {model} | h={int(h)} | {mase_s} | {_fmt(g['mae'].mean())} | "
            f"{_fmt(g['rmse'].mean())} | {n} |"
        )
    return "\n".join(lines)


def reproduction_table(cmp: pd.DataFrame) -> str:
    lines = [
        "| Reference | Model | Horizon | Published MASE | Reproduced MASE | Rel. diff | "
        "Max fold diff | Tolerance | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cmp.itertuples():
        lines.append(
            f"| `{r.reference_file}` | {r.model} | h={int(r.horizon)} | {_fmt(r.reference)} | "
            f"{_fmt(r.reproduced)} | {r.rel_diff:.2%} | {_fmt(r.max_fold_diff)} | "
            f"{r.tolerance:.0%} | {r.status} |"
        )
    return "\n".join(lines)


def render_summary(cfg: BenchmarkConfig, report: EvaluationReport) -> str:
    parts = [f"# Benchmark `{cfg.name}`", "", cfg.description.strip(), ""]
    if report.reproduction is not None:
        cmp = report.reproduction
        counts = cmp["status"].value_counts().to_dict()
        verdict = "PASS" if report.reproduced else "FAIL"
        parts += [
            "## Reproduction vs published results",
            "",
            f"Overall: **{verdict}** - "
            + ", ".join(f"{counts.get(s, 0)} {s}" for s in ("PASS", "KNOWN", "FAIL", "MISSING")),
            "",
            reproduction_table(cmp),
            "",
        ]
        notes = cmp[cmp["status"] == "KNOWN"].drop_duplicates(["reference_file", "model"])
        if len(notes):
            parts += [
                "**Known differences** (published numbers the committed code cannot "
                "produce; evidence in RESULTS.md):",
                "",
            ]
            parts += [f"- `{r.reference_file}` / {r.model}: {r.note}" for r in notes.itertuples()]
            parts += [""]
    for name, table in report.run_tables.items():
        parts += [f"## Run `{name}`", "", results_table(table), ""]
    if report.missing:
        parts += ["## Missing forecasts", "", *[f"- {m}" for m in report.missing], ""]
    parts += [
        "MAE/RMSE are in standardised units (z-scores of the training fold), as in the "
        "original study; MASE is scale-free.",
        "",
    ]
    return "\n".join(parts)
