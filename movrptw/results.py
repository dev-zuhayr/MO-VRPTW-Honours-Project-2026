"""Aggregate :class:`SolverResult` objects and write a multi-sheet workbook.

Sheets produced:

* ``runs``      - one row per replication (raw), including hypervolume
* ``summary``   - aggregated mean/std/min/max per instance x algorithm x config,
                  success rate, gap-to-BKS and mean/std hypervolume
* ``pareto``    - every point of every multi-objective Pareto front
* ``reference`` - the HV reference point and normalisation used per instance
* ``solutions`` - the best solution's routes per run (as a string)
* ``instances`` - instance metadata
* ``config``    - the experiment configuration used

Set ``append=True`` to merge into an existing workbook instead of overwriting.

Note on append + hypervolume: HV is relative to the normalisation pooled from
the runs written in one pass (see :mod:`movrptw.metrics`). Rows merged in from
an existing workbook keep the values they were written with, which came from a
different ideal/nadir, so they are not comparable with the new rows. Rerun
without ``append`` (the checkpoints make that cheap) whenever HV is the point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from . import bks
from . import metrics as metrics_mod
from .config import ExperimentConfig
from .instance import Instance
from .solvers.base import SolverResult


def _size_from_cfg(cfg: Optional[ExperimentConfig]) -> Optional[int]:
    """``bks.size_from_config(cfg)``, or ``None`` if ``cfg`` is absent or has
    no size field set (e.g. a full-size run with ``max_customers`` unset —
    ``bks.gap``/``bks.lookup`` already fall back sanely on ``size=None``).
    """
    if cfg is None:
        return None
    try:
        return bks.size_from_config(cfg)
    except KeyError:
        return None

def runs_frame(
    results: Iterable[SolverResult],
    quality: Optional[Sequence[metrics_mod.FrontQuality]] = None,
    cfg: Optional[ExperimentConfig] = None,
) -> pd.DataFrame:
    """One row per replication. ``quality`` is computed if not supplied.

    ``cfg``, if given, supplies the instance size via ``bks.size_from_config``
    so the BKS gap is resolved reliably instead of via the instance name's
    (fragile) size suffix. Without a ``cfg`` — or if it has no size field set,
    e.g. a full-size run with ``max_customers`` unset — falls back to the
    suffix-based/100-customer-table resolution in ``bks.gap`` itself.
    """
    results = list(results)
    if quality is None:
        quality, _ = metrics_mod.run_metrics(results)
    size = _size_from_cfg(cfg)
    rows = []
    for r, q in zip(results, quality):
        row = r.summary_row()
        row.update(
            bks.gap(
                r.instance,
                int(r.best.objectives.get("num_vehicles", 0)),
                float(r.best.objectives.get("total_distance", 0.0)),
                size=size,
            )
        )
        row["hv"] = round(q.hv, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_frame(runs: pd.DataFrame, objective_names: List[str]) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    group_keys = ["instance", "algorithm", "params"]
    agg: Dict[str, list] = {}
    for obj in objective_names:
        if obj in runs.columns:
            agg[obj] = ["mean", "std", "min", "max"]
    agg["runtime_s"] = ["mean", "std"]
    agg["feasible"] = ["mean"]  # feasibility = success rate
    # Front quality: mean/std across replications. NaN (no admissible front)
    # is skipped by pandas rather than counted as zero.
    if "hv" in runs.columns:
        agg["hv"] = ["mean", "std"]

    summary = runs.groupby(group_keys, dropna=False).agg(agg)
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary = summary.rename(columns={"feasible_mean": "success_rate"})
    summary = summary.reset_index()

    # Attach BKS gap for the best (min-distance) run within each group.
    if "gap_distance_pct" in runs.columns:
        best_gap = (
            runs.sort_values("total_distance")
            .groupby(group_keys, dropna=False)[["gap_vehicles_pct", "gap_distance_pct"]]
            .first()
            .reset_index()
        )
        summary = summary.merge(best_gap, on=group_keys, how="left")
    return summary


def pareto_frame(results: Iterable[SolverResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for k, pt in enumerate(r.pareto):
            row = {
                "instance": r.instance,
                "algorithm": r.algorithm,
                "seed": r.seed,
                "point": k,
            }
            row.update(pt.objectives)
            row["routes"] = _routes_str(pt.routes)
            rows.append(row)
    return pd.DataFrame(rows)


def reference_frame(
    groups: Dict[Tuple[str, Tuple[str, ...]], metrics_mod.FrontComparison],
) -> pd.DataFrame:
    """Document how HV was measured, one row per comparison group.

    Without this the metric is an unreproducible number: it only means something
    relative to the reference point and the ideal/nadir scaling of the group it
    was computed in.
    """
    rows = []
    for (instance, names), comp in sorted(groups.items()):
        row = {
            "instance": instance,
            "objectives": ", ".join(names),
            "n_runs": len(comp.quality),
            "hv_ref_point": comp.ref_factor,  # normalised space; nadir = 1.0
        }
        for j, name in enumerate(names):
            row[f"ideal_{name}"] = round(float(comp.bounds.ideal[j]), 4)
            row[f"nadir_{name}"] = round(float(comp.bounds.nadir[j]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def solutions_frame(results: Iterable[SolverResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "instance": r.instance,
                "algorithm": r.algorithm,
                "seed": r.seed,
                "feasible": r.best.evaluation.feasible,
                "num_vehicles": r.best.objectives.get("num_vehicles"),
                "total_distance": round(r.best.objectives.get("total_distance", 0.0), 3),
                "routes": _routes_str(r.best.routes),
            }
        )
    return pd.DataFrame(rows)


def instances_frame(
    instances: Iterable[Instance],
    cfg: Optional[ExperimentConfig] = None,
) -> pd.DataFrame:
    size = _size_from_cfg(cfg)
    rows = []
    for ins in instances:
        bk = bks.lookup(ins.name, size=size)
        rows.append(
            {
                "instance": ins.name,
                "n_customers": ins.n_customers,
                "fleet_size": ins.num_vehicles,
                "capacity": ins.capacity,
                "horizon": ins.horizon,
                "total_demand": float(ins.demands.sum()),
                "bks_vehicles": bk[0] if bk else None,
                "bks_distance": bk[1] if bk else None,
            }
        )
    return pd.DataFrame(rows)


def config_frame(cfg: ExperimentConfig) -> pd.DataFrame:
    items = {
        "objectives": ", ".join(cfg.objectives),
        "n_runs": cfg.n_runs,
        "base_seed": cfg.base_seed,
        "n_jobs": cfg.n_jobs,
        "timeout_s": cfg.timeout_s,
        "distance_rounding": cfg.distance_rounding,
        "algorithms": ", ".join(a.name for a in cfg.algorithms),
        "n_instances": len(cfg.instances),
    }
    return pd.DataFrame([{"key": k, "value": v} for k, v in items.items()])


def _routes_str(routes: List[List[int]]) -> str:
    return " | ".join("0-" + "-".join(map(str, r)) + "-0" for r in routes)


def write_workbook(
    path: str | Path,
    *,
    results: List[SolverResult],
    instances: List[Instance],
    cfg: ExperimentConfig,
    append: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    objective_names = cfg.objectives
    quality, groups = metrics_mod.run_metrics(results)
    frames = {
        "runs": runs_frame(results, quality, cfg),
        "pareto": pareto_frame(results),
        "reference": reference_frame(groups),
        "solutions": solutions_frame(results),
        "instances": instances_frame(instances, cfg),
        "config": config_frame(cfg),
    }

    if append and path.exists():
        frames = _merge_existing(path, frames)

    # Summary is recomputed from the (possibly merged) runs frame.
    frames["summary"] = summary_frame(frames["runs"], objective_names)

    order = ["summary", "runs", "pareto", "reference", "solutions", "instances", "config"]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in order:
            df = frames.get(sheet, pd.DataFrame())
            if df is None or df.empty:
                df = pd.DataFrame({"info": [f"(no {sheet} data)"]})
            df.to_excel(writer, sheet_name=sheet, index=False)
            _format_sheet(writer.sheets[sheet], df)
    return path


def _merge_existing(path: Path, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    try:
        existing = pd.read_excel(path, sheet_name=None)
    except Exception:
        return frames
    for sheet in ("runs", "pareto", "solutions", "instances"):
        old = existing.get(sheet)
        if old is not None and not old.empty:
            frames[sheet] = pd.concat([old, frames[sheet]], ignore_index=True)
    if "instances" in frames:
        frames["instances"] = frames["instances"].drop_duplicates(subset=["instance"])
    return frames


def _format_sheet(ws, df: pd.DataFrame) -> None:
    """Light cosmetic formatting: bold header, frozen top row, autosized cols."""
    from openpyxl.styles import Font

    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for col_idx, col in enumerate(df.columns, start=1):
        max_len = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)])
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 60)
