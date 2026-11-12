"""Solver framework: abstract base class, result containers, registry.

To add a new metaheuristic:
1. Subclass MetaheuristicSolver and implement _solve.
2. Decorate it with @register_solver("my_algo").
3. Reference "my_algo" from a config file.

Replications, seeding, timing, and Pareto extraction are handled here.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from ..instance import Instance
from ..solution import Evaluation, Solution, evaluate


@dataclass
class ScoredSolution:
    """A solution paired with its evaluation and objective vector."""

    solution: Solution
    evaluation: Evaluation
    objectives: Dict[str, float]

    @property
    def routes(self) -> List[List[int]]:
        return self.solution.active_routes


@dataclass
class SolverResult:
    """Standardised result of a single solver run (one replication)."""

    algorithm: str
    instance: str
    seed: int
    params: Dict
    objective_names: List[str]
    best: ScoredSolution
    pareto: List[ScoredSolution] = field(default_factory=list)
    runtime_s: float = 0.0
    iterations: int = 0
    convergence: List[float] = field(default_factory=list)
    feasible: bool = True

    def summary_row(self) -> Dict:
        """Flatten the run into a single dict suitable for a DataFrame row."""
        row = {
            "instance": self.instance,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "runtime_s": round(self.runtime_s, 4),
            "iterations": self.iterations,
            "feasible": self.feasible,
            "n_pareto": len(self.pareto),
        }
        for name in self.objective_names:
            row[name] = self.best.objectives.get(name)
        row["params"] = _short_params(self.params)
        return row


def _short_params(params: Dict) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(params.items()))


def make_scalariser(
    instance: Instance,
    objective_names: Sequence[str],
    weights: Optional[Sequence[float]] = None,
) -> Callable[[Evaluation], float]:
    """Map an Evaluation to a single penalised scalar: objectives normalised
    by instance-scale factors and combined with weights, plus a heavy
    infeasibility penalty to pull search back into the feasible region."""
    names = list(objective_names)
    if weights is None:
        weights = [1.0] * len(names)
    scale = _objective_scales(instance)
    w = np.asarray(weights, dtype=float)

    def scalar(ev: Evaluation) -> float:
        vec = np.array([ev.objective(n) / scale[n] for n in names])
        penalty = 1e4 * (ev.capacity_violation / max(instance.capacity, 1.0)
                         + ev.time_window_violation / max(instance.horizon, 1.0)
                         + ev.unserved)
        return float((w * vec).sum() + penalty)

    return scalar


def _objective_scales(instance: Instance) -> Dict[str, float]:
    diag = float(np.hypot(np.ptp(instance.coords[:, 0]), np.ptp(instance.coords[:, 1]))) or 1.0
    n = max(instance.n_customers, 1)
    return {
        "num_vehicles": float(max(instance.num_vehicles, 1)),
        "total_distance": diag * n,
        "total_wait": diag * n,
        "total_tardiness": diag * n,
        "route_imbalance": diag,
        "makespan": max(instance.horizon, 1.0),
    }


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if `a` Pareto-dominates `b` (all <=, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated(scored: Sequence[ScoredSolution], names: Sequence[str]) -> List[ScoredSolution]:
    """Extract the non-dominated front, deduplicating identical objective vectors."""
    vecs = [np.array([s.objectives[n] for n in names]) for s in scored]
    keep: List[ScoredSolution] = []
    for i, si in enumerate(scored):
        if not any(dominates(vecs[j], vecs[i]) for j in range(len(scored)) if j != i):
            if not any(np.array_equal(vecs[i], np.array([k.objectives[n] for n in names])) for k in keep):
                keep.append(si)
    return keep


class MetaheuristicSolver(ABC):
    """Base class for all metaheuristic solvers.

    Subclasses implement _solve and return a list of candidate Solutions
    representing their final population/archive. This class evaluates them,
    builds the Pareto front (multi-objective) or scalarised best
    (single-objective), and packages a SolverResult.
    """

    name: str = "base"
    multi_objective: bool = False

    def __init__(
        self,
        instance: Instance,
        objective_names: Sequence[str],
        params: Optional[Dict] = None,
        *,
        seed: int = 0,
    ) -> None:
        self.instance = instance
        self.objective_names = list(objective_names)
        self.params = dict(params or {})
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.dist = instance.distance_matrix()
        self._convergence: List[float] = []
        self._iterations = 0
        self.scalar = make_scalariser(
            instance, self.objective_names, self.params.get("weights")
        )

    def evaluate(self, solution: Solution) -> Evaluation:
        return evaluate(solution, self.instance, self.dist)

    def score(self, solution: Solution) -> ScoredSolution:
        ev = self.evaluate(solution)
        objs = {n: ev.objective(n) for n in self.objective_names}
        return ScoredSolution(solution=solution, evaluation=ev, objectives=objs)

    @abstractmethod
    def _solve(self) -> List[Solution]:
        """Run the search and return the final pool of candidate solutions."""
        raise NotImplementedError

    def run(self) -> SolverResult:
        start = time.perf_counter()
        pool = self._solve()
        runtime = time.perf_counter() - start

        scored = [self.score(s) for s in pool] or [self.score(Solution([]))]
        if self.multi_objective and len(self.objective_names) > 1:
            pareto = non_dominated(scored, self.objective_names)
            best = min(scored, key=lambda s: self.scalar(s.evaluation))
        else:
            pareto = []
            best = min(scored, key=lambda s: self.scalar(s.evaluation))

        return SolverResult(
            algorithm=self.name,
            instance=self.instance.name,
            seed=self.seed,
            params=self.params,
            objective_names=self.objective_names,
            best=best,
            pareto=pareto,
            runtime_s=runtime,
            iterations=self._iterations,
            convergence=self._convergence,
            feasible=best.evaluation.feasible,
        )


_REGISTRY: Dict[str, Type[MetaheuristicSolver]] = {}


def register_solver(name: str) -> Callable[[Type[MetaheuristicSolver]], Type[MetaheuristicSolver]]:
    def deco(cls: Type[MetaheuristicSolver]) -> Type[MetaheuristicSolver]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_solver(name: str) -> Type[MetaheuristicSolver]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown solver '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_solvers() -> List[str]:
    return sorted(_REGISTRY)
