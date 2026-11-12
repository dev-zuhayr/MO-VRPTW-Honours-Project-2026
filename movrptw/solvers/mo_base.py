"""Shared machinery for the multi-objective solvers (SPEA2, MOACO).

Handles: reading tw_mode and validating it against the objective list; the
admissibility gate (capacity + coverage always hard; time windows hard only
in hard mode); building an (n, m) objective matrix for a pool of solutions
(m = 2 in hard mode, 3 in soft); and a Pareto archive with non-dominated
insertion. Objective count is read from self.objective_names and never
hardcoded, so both algorithms handle hard/soft runs from one implementation.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .. import tw
from ..instance import Instance
from ..solution import Solution
from .base import MetaheuristicSolver, dominates


class MOSolver(MetaheuristicSolver):
    """Base for Pareto-based solvers with hard/soft time-window support."""

    multi_objective = True

    # Large finite penalty added to every objective of an inadmissible
    # solution, so infeasible individuals are always dominated by feasible
    # ones but can still be ranked against each other.
    PENALTY = 1e6

    def __init__(self, instance: Instance, objective_names, params=None, *, seed: int = 0):
        super().__init__(instance, objective_names, params, seed=seed)
        self.tw_mode = str(self.params.get("tw_mode", tw.SOFT)).lower()
        tw.validate(self.tw_mode, self.objective_names)
        self.respect_tw = self.tw_mode == tw.HARD

    def is_admissible(self, sol: Solution) -> bool:
        return tw.admissible(self.evaluate(sol), self.tw_mode)

    def violation(self, sol: Solution) -> float:
        return tw.constraint_violation(self.evaluate(sol), self.tw_mode)

    def objective_matrix(self, pool: Sequence[Solution]) -> np.ndarray:
        """(len(pool), len(objective_names)) matrix of objectives. Inadmissible
        solutions get PENALTY * (1 + violation) added to every coordinate, so
        any admissible solution dominates them while a gradient is preserved
        among the infeasible ones. Tardiness itself is never penalised here —
        in soft mode it's a genuine objective, and hard mode already blocks it
        via the admissibility gate."""
        m = len(self.objective_names)
        objs = np.zeros((len(pool), m), dtype=float)
        for i, sol in enumerate(pool):
            ev = self.evaluate(sol)
            objs[i] = ev.vector(self.objective_names)
            if not tw.admissible(ev, self.tw_mode):
                objs[i] += self.PENALTY * (1.0 + tw.constraint_violation(ev, self.tw_mode))
        return objs

    def update_archive(
        self,
        archive: List[Solution],
        candidates: Sequence[Solution],
        *,
        max_size: int | None = None,
    ) -> List[Solution]:
        """Insert admissible candidates, keeping only non-dominated solutions
        (A <- {s in A u C : not exists s' in A u C, s' > s}), collapsing
        duplicate objective vectors. Thins via _thin if max_size is exceeded."""
        pool = list(archive) + [s for s in candidates if self.is_admissible(s)]
        if not pool:
            return []

        objs = self.objective_matrix(pool)
        keep: List[int] = []
        seen: set = set()
        for i in range(len(pool)):
            if any(dominates(objs[j], objs[i]) for j in range(len(pool)) if j != i):
                continue
            key = tuple(np.round(objs[i], 6))
            if key in seen:
                continue
            seen.add(key)
            keep.append(i)

        new_archive = [pool[i] for i in keep]
        if max_size is not None and len(new_archive) > max_size:
            new_archive = self._thin(new_archive, max_size)
        return new_archive

    def _thin(self, archive: List[Solution], max_size: int) -> List[Solution]:
        """Generic crowding-based thinning: drop the most crowded repeatedly."""
        objs = normalise(self.objective_matrix(archive))
        alive = list(range(len(archive)))
        while len(alive) > max_size:
            sub = objs[alive]
            d = pairwise_distances(sub)
            np.fill_diagonal(d, np.inf)
            nearest = d.min(axis=1)
            alive.pop(int(np.argmin(nearest)))
        return [archive[i] for i in alive]

    def log_generation(self, pool: Sequence[Solution]) -> None:
        self._iterations += 1
        if pool:
            self._convergence.append(
                float(min(self.scalar(self.evaluate(s)) for s in pool))
            )


def normalise(objs: np.ndarray) -> np.ndarray:
    """Scale each objective column to [0, 1] — needed before any distance
    computation, since e.g. total_distance (hundreds) would otherwise swamp
    num_vehicles (single digits) and make crowding/density meaningless."""
    lo = objs.min(axis=0)
    hi = objs.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (objs - lo) / span


def pairwise_distances(objs: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between rows of `objs`."""
    diff = objs[:, None, :] - objs[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))
