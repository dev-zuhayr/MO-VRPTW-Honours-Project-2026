"""Example: adding your own solver in ~40 lines.

This Tabu Search is **not** registered by default. It shows the complete
contract for plugging a new algorithm into the framework:

1. Subclass ``MetaheuristicSolver``.
2. Implement ``_solve()`` returning a list of candidate ``Solution`` objects.
3. Decorate with ``@register_solver("name")``.

Once imported anywhere before an experiment runs, ``"tabu"`` becomes usable
from a config file exactly like the built-in solvers. To activate it, either
import this module in your run script, or move it into ``movrptw/solvers/`` and
add it to that package's ``__init__``.

    from examples.custom_solver import TabuSearch  # noqa: registers "tabu"
"""

from __future__ import annotations

from collections import deque
from typing import List

from movrptw.solution import Solution
from movrptw.solvers.base import MetaheuristicSolver, register_solver
from movrptw.solvers.construction import bind_operators, nearest_neighbour_construct


@register_solver("tabu")
class TabuSearch(MetaheuristicSolver):
    multi_objective = False
    DEFAULTS = dict(iterations=4000, tabu_tenure=25, neighbours_per_step=20)

    def _p(self, key):
        return self.params.get(key, self.DEFAULTS[key])

    def _solve(self) -> List[Solution]:
        iters = int(self._p("iterations"))
        tenure = int(self._p("tabu_tenure"))
        n_neigh = int(self._p("neighbours_per_step"))
        operators = bind_operators(self.instance, self.dist)

        current = nearest_neighbour_construct(self.instance, self.dist, self.rng)
        best, best_f = current.copy(), self.scalar(self.evaluate(current))
        tabu: deque = deque(maxlen=tenure)

        for _ in range(iters):
            # Sample a neighbourhood and pick the best non-tabu candidate.
            candidates = []
            for _ in range(n_neigh):
                op = self.rng.choice(operators)
                cand = op(current, self.rng)
                key = self._signature(cand)
                if key in tabu:
                    continue
                candidates.append((self.scalar(self.evaluate(cand)), key, cand))
            if not candidates:
                continue
            candidates.sort(key=lambda t: t[0])
            f, key, current = candidates[0]
            tabu.append(key)
            if f < best_f:
                best, best_f = current.copy(), f
            self._iterations += 1
            self._convergence.append(best_f)

        return [best]

    @staticmethod
    def _signature(sol: Solution) -> tuple:
        return tuple(tuple(r) for r in sol.active_routes)
