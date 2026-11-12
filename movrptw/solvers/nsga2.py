"""NSGA-II for the multi-objective VRPTW — hard and soft time windows.

Two additions on top of plain NSGA-II:
1. Route-split operator (_split): the only move that can increase vehicle
   count, needed so the population can explore that side of the trade-off.
2. Diversity injection: caps how many individuals may share a near-identical
   objective vector, replacing the excess with fresh random solutions so
   the population doesn't collapse onto one point.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from ..solution import Solution
from .base import dominates, register_solver
from .construction import (
    bind_operators,
    nearest_neighbour_construct,
    random_neighbour,
    route_crossover,
)
from .mo_base import MOSolver


def _split(solution: Solution, rng: random.Random) -> Solution:
    """Split one randomly chosen route (>= 2 customers) into two."""
    s = solution.copy()
    candidates = [i for i, r in enumerate(s.routes) if len(r) >= 2]
    if not candidates:
        return s
    idx = rng.choice(candidates)
    route = s.routes[idx]
    cut = rng.randrange(1, len(route))
    s.routes[idx:idx + 1] = [route[:cut], route[cut:]]
    return s


def _diversity_injection(
    pop: List[Solution],
    objs: np.ndarray,
    max_duplicates: int,
    instance,
    dist: np.ndarray,
    rng: random.Random,
    respect_tw: bool,
) -> List[Solution]:
    """Cap duplicate objective vectors, replacing excess with fresh solutions."""
    seen: Dict[Tuple[float, ...], int] = {}
    keep_idx: List[int] = []
    n_replace = 0

    for i in range(len(pop)):
        key = tuple(round(float(v), 3) for v in objs[i])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_duplicates:
            keep_idx.append(i)
        else:
            n_replace += 1

    new_pop = [pop[i] for i in keep_idx]
    for _ in range(n_replace):
        new_pop.append(
            nearest_neighbour_construct(instance, dist, rng, respect_tw=respect_tw)
        )
    return new_pop


@register_solver("nsga2")
class NSGA2(MOSolver):

    DEFAULTS = dict(
        pop_size=100,
        generations=200,
        crossover_rate=1,
        mutation_rate=1,
        max_duplicates=3,
        tw_mode="hard",
    )

    def __init__(self, instance, objective_names, params=None, *, seed: int = 0):
        params = dict(params or {})
        params.setdefault("tw_mode", self.DEFAULTS["tw_mode"])
        super().__init__(instance, objective_names, params, seed=seed)

    def _p(self, key: str):
        return self.params.get(key, self.DEFAULTS[key])

    def _fast_non_dominated_sort(self, objs: np.ndarray) -> List[List[int]]:
        n = len(objs)
        S = [[] for _ in range(n)]
        ndom = [0] * n
        fronts: List[List[int]] = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if dominates(objs[p], objs[q]):
                    S[p].append(q)
                elif dominates(objs[q], objs[p]):
                    ndom[p] += 1
            if ndom[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    ndom[q] -= 1
                    if ndom[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    @staticmethod
    def _crowding(front: List[int], objs: np.ndarray) -> dict:
        dist = {i: 0.0 for i in front}
        if len(front) <= 2:
            return {i: float("inf") for i in front}
        m = objs.shape[1]
        for k in range(m):
            order = sorted(front, key=lambda i: objs[i, k])
            dist[order[0]] = dist[order[-1]] = float("inf")
            lo, hi = objs[order[0], k], objs[order[-1], k]
            span = hi - lo or 1.0
            for idx in range(1, len(order) - 1):
                dist[order[idx]] += (objs[order[idx + 1], k] - objs[order[idx - 1], k]) / span
        return dist

    def _make_offspring(self, pop, ranks, crowd, operators) -> List[Solution]:
        def tourney() -> Solution:
            a, b = self.rng.randrange(len(pop)), self.rng.randrange(len(pop))
            if ranks[a] != ranks[b]:
                return pop[a] if ranks[a] < ranks[b] else pop[b]
            return pop[a] if crowd[a] >= crowd[b] else pop[b]

        cx_rate = float(self._p("crossover_rate"))
        mut_rate = float(self._p("mutation_rate"))
        children: List[Solution] = []
        while len(children) < len(pop):
            p1, p2 = tourney(), tourney()
            child = (
                route_crossover(
                    p1, p2, self.instance, self.dist, self.rng,
                    respect_tw=self.respect_tw,
                )
                if self.rng.random() < cx_rate
                else p1.copy()
            )
            if self.rng.random() < mut_rate:
                child = random_neighbour(child, self.rng, operators)
            children.append(child)
        return children

    def _solve(self) -> List[Solution]:
        pop_size = int(self._p("pop_size"))
        generations = int(self._p("generations"))
        max_duplicates = int(self._p("max_duplicates"))

        operators = bind_operators(self.instance, self.dist, respect_tw=self.respect_tw) + (_split,)

        pop = [
            nearest_neighbour_construct(
                self.instance, self.dist, self.rng, respect_tw=self.respect_tw
            )
            for _ in range(pop_size)
        ]

        for _ in range(generations):
            objs = self.objective_matrix(pop)
            fronts = self._fast_non_dominated_sort(objs)
            ranks = {i: r for r, f in enumerate(fronts) for i in f}
            crowd: dict = {}
            for f in fronts:
                crowd.update(self._crowding(f, objs))

            offspring = self._make_offspring(pop, ranks, crowd, operators)

            # Environmental selection over combined parent + offspring pool.
            combined = pop + offspring
            objs_c = self.objective_matrix(combined)
            fronts_c = self._fast_non_dominated_sort(objs_c)

            new_pop: List[Solution] = []
            new_objs: List[np.ndarray] = []
            for f in fronts_c:
                if len(new_pop) + len(f) <= pop_size:
                    new_pop.extend(combined[i] for i in f)
                    new_objs.extend(objs_c[i] for i in f)
                else:
                    cd = self._crowding(f, objs_c)
                    for i in sorted(f, key=lambda i: cd[i], reverse=True):
                        if len(new_pop) >= pop_size:
                            break
                        new_pop.append(combined[i])
                        new_objs.append(objs_c[i])
                    break

            new_pop = _diversity_injection(
                new_pop, np.array(new_objs), max_duplicates,
                self.instance, self.dist, self.rng, self.respect_tw,
            )
            pop = new_pop

            self.log_generation(pop)

        return pop
