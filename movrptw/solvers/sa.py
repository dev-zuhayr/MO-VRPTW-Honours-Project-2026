"""Simulated Annealing for VRPTW (single-objective via scalarisation).

A textbook geometric-cooling SA over the shared neighbourhood operators. Useful
as a fast baseline and as a second worked example of the single-objective
plug-in pattern.
"""

from __future__ import annotations

import math
from typing import List

from ..solution import Solution
from .base import MetaheuristicSolver, register_solver
from .construction import bind_operators, nearest_neighbour_construct, random_neighbour


@register_solver("sa")
class SimulatedAnnealing(MetaheuristicSolver):

    multi_objective = False

    DEFAULTS = dict(
        iterations=8000,
        initial_temp=1.0,
        cooling=0.9995,
        restarts=1,
    )

    def _p(self, key: str):
        return self.params.get(key, self.DEFAULTS[key])

    def _solve(self) -> List[Solution]:
        iterations = int(self._p("iterations"))
        t0 = float(self._p("initial_temp"))
        cooling = float(self._p("cooling"))
        restarts = int(self._p("restarts"))

        operators = bind_operators(self.instance, self.dist)

        global_best = None
        global_best_f = math.inf

        for _ in range(restarts):
            current = nearest_neighbour_construct(self.instance, self.dist, self.rng)
            current_f = self.scalar(self.evaluate(current))
            best, best_f = current.copy(), current_f
            temp = t0

            for _ in range(iterations):
                candidate = random_neighbour(current, self.rng, operators)
                cand_f = self.scalar(self.evaluate(candidate))
                delta = cand_f - current_f
                if delta < 0 or self.rng.random() < math.exp(-delta / max(temp, 1e-12)):
                    current, current_f = candidate, cand_f
                    if current_f < best_f:
                        best, best_f = current.copy(), current_f
                temp *= cooling
                self._iterations += 1
                self._convergence.append(best_f)

            if best_f < global_best_f:
                global_best, global_best_f = best, best_f

        return [global_best] if global_best is not None else []
