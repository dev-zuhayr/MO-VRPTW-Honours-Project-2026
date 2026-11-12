"""Config-driven batch experiment runner.

Expands ``instances x algorithms x param-grid x seeds`` into a flat task list,
runs each task (optionally in parallel with joblib), checkpoints partial
results to disk, then writes the aggregated workbook.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from joblib import Parallel, delayed

from . import results as results_mod
from .config import ExperimentConfig, load_config
from .instance import Instance, parse_solomon
from .solvers.base import SolverResult, get_solver

logger = logging.getLogger("movrptw")

def setup_logging(logfile: Optional[str | Path] = None, level: int = logging.INFO) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=handlers,
        force=True,
    )

@dataclass
class Task:
    instance_path: str
    algorithm: str
    params: Dict
    seed: int
    objectives: List[str]
    distance_rounding: Optional[str]
    max_customers: Optional[int] = None

    @property
    def key(self) -> str:
        pstr = ";".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{Path(self.instance_path).stem}|{self.algorithm}|{pstr}|seed={self.seed}"


def build_tasks(cfg: ExperimentConfig) -> List[Task]:
    tasks: List[Task] = []
    for inst_path in cfg.instances:
        for algo in cfg.algorithms:
            for params in algo.param_sets():
                for run in range(cfg.n_runs):
                    tasks.append(
                        Task(
                            instance_path=inst_path,
                            algorithm=algo.name,
                            params=params,
                            seed=cfg.base_seed + run,
                            objectives=cfg.objectives,
                            distance_rounding=cfg.distance_rounding,
                            max_customers=cfg.max_customers,
                        )
                    )
    return tasks

_INSTANCE_CACHE: Dict[str, Instance] = {}


def _load_instance(path: str, rounding: Optional[str], max_customers: Optional[int] = None) -> Instance:
    cache_key = (path, max_customers)
    if cache_key not in _INSTANCE_CACHE:
        ins = parse_solomon(path)
        if max_customers is not None and ins.n_customers > max_customers:
            ins = Instance(
                name=ins.name,
                nodes=[ins.depot] + ins.customers[:max_customers],
                num_vehicles=ins.num_vehicles,
                capacity=ins.capacity,
            )
        ins.distance_matrix(rounding=rounding)
        _INSTANCE_CACHE[cache_key] = ins
    return _INSTANCE_CACHE[cache_key]


def run_task(task: Task) -> SolverResult:
    instance = _load_instance(task.instance_path, task.distance_rounding, task.max_customers)
    solver_cls = get_solver(task.algorithm)
    solver = solver_cls(instance, task.objectives, task.params, seed=task.seed)
    return solver.run()


def _run_task_safe(task: Task, checkpoint_dir: Optional[Path]) -> Optional[SolverResult]:
    t0 = time.perf_counter()
    try:
        res = run_task(task)
        logger.info(
            "done  %-55s vehicles=%s dist=%.1f feasible=%s (%.1fs)",
            task.key,
            res.best.objectives.get("num_vehicles"),
            res.best.objectives.get("total_distance", float("nan")),
            res.feasible,
            time.perf_counter() - t0,
        )
        if checkpoint_dir is not None:
            _checkpoint(checkpoint_dir, task, res)
        return res
    except Exception as exc:  # keep the batch alive if a single task fails
        logger.exception("FAILED %s: %s", task.key, exc)
        return None


def _checkpoint(checkpoint_dir: Path, task: Task, res: SolverResult) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe = task.key.replace("|", "__").replace("=", "-").replace(";", "_")
    with open(checkpoint_dir / f"{safe}.pkl", "wb") as fh:
        pickle.dump(res, fh)


def _load_checkpoints(checkpoint_dir: Path) -> Dict[str, SolverResult]:
    done: Dict[str, SolverResult] = {}
    if not checkpoint_dir.exists():
        return done
    for f in checkpoint_dir.glob("*.pkl"):
        try:
            with open(f, "rb") as fh:
                res: SolverResult = pickle.load(fh)
            done[f.stem] = res
        except Exception:
            continue
    return done

def run_experiment(
    cfg: ExperimentConfig,
    *,
    resume: bool = True,
    logfile: Optional[str | Path] = None,
) -> List[SolverResult]:
    setup_logging(logfile)
    cfg.validate()
    checkpoint_dir = Path(cfg.checkpoint_dir)

    tasks = build_tasks(cfg)
    logger.info("Built %d tasks across %d instances and %d algorithms.",
                len(tasks), len(cfg.instances), len(cfg.algorithms))

    existing = _load_checkpoints(checkpoint_dir) if resume else {}
    if existing:
        logger.info("Resuming: %d completed tasks found in checkpoints.", len(existing))
    done_stems = set(existing)

    def safe_stem(task: Task) -> str:
        return task.key.replace("|", "__").replace("=", "-").replace(";", "_")

    pending = [t for t in tasks if safe_stem(t) not in done_stems]
    logger.info("%d tasks pending.", len(pending))

    if pending:
        Parallel(n_jobs=cfg.n_jobs, backend="loky")(
            delayed(_run_task_safe)(t, checkpoint_dir) for t in pending
        )

    # Reload everything from the checkpoints so the results are complete and
    # consistent, whether they came from this run or a previous (resumed) one.
    all_done = _load_checkpoints(checkpoint_dir)
    wanted = {safe_stem(t) for t in tasks}
    results = [all_done[s] for s in wanted if s in all_done]

    instances = [_load_instance(p, cfg.distance_rounding, cfg.max_customers) for p in cfg.instances]
    out = results_mod.write_workbook(
        cfg.output, results=results, instances=instances, cfg=cfg, append=cfg.append
    )
    logger.info("Wrote workbook: %s (%d runs).", out, len(results))
    return results


def run_from_config_file(path: str | Path, **kwargs) -> List[SolverResult]:
    return run_experiment(load_config(path), **kwargs)
