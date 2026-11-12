"""GRASP (Greedy Randomized Adaptive Search Procedure) for the multi-objective
VRPTW with hard and soft time windows.

Each iteration: 
(1) Construction builds one solution greedily but randomized
via a Restricted Candidate List.
(2) Local search hill-climbs it with relocate, swap, 2-opt, and the shared
route-elimination move. relocate/swap only consider each customer's
k-nearest neighbours (neighbor_k) rather than every route/position.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np

from ..solution import Solution, evaluate_route
from .base import register_solver
from .construction import bind_operators, feasible_insert_cost
from .mo_base import MOSolver


def _construct(instance, dist: np.ndarray, alpha: float, rng: random.Random,
                respect_tw: bool) -> Solution:
    """Build one solution by repeatedly extending the current route from an
    RCL of cheap-to-append candidates, starting a new route once nothing
    fits. Capacity is always enforced; time windows only when respect_tw."""
    unrouted = set(range(1, instance.size))
    routes: List[List[int]] = []

    while unrouted:
        route: List[int] = []
        while True:
            candidates = []
            for c in unrouted:
                cost = feasible_insert_cost(
                    route, len(route), c, instance, dist, respect_tw=respect_tw
                )
                if cost is not None:
                    candidates.append((c, cost))
            if not candidates:
                break

            costs = [cost for _, cost in candidates]
            best, worst = min(costs), max(costs)
            threshold = best + alpha * (worst - best)
            rcl = [c for c, cost in candidates if cost <= threshold]

            chosen = rng.choice(rcl)
            route.append(chosen)
            unrouted.discard(chosen)

        if route:
            routes.append(route)
        elif unrouted:
            # Nothing fits this customer at all under the current mode —
            # force it into its own route so construction terminates.
            routes.append([unrouted.pop()])

    return Solution(routes)


def _nearest_neighbor_lists(instance, dist: np.ndarray, k: int) -> List[List[int]]:
    """result[c] = c's k nearest other customers by distance, nearest first."""
    n = instance.size
    neighbours: List[List[int]] = [[] for _ in range(n)]
    for c in range(1, n):
        order = sorted((j for j in range(1, n) if j != c), key=lambda j: dist[c][j])
        neighbours[c] = order[:k]
    return neighbours


def _locate_customers(routes: List[List[int]]) -> Dict[int, Tuple[int, int]]:
    loc: Dict[int, Tuple[int, int]] = {}
    for ri, route in enumerate(routes):
        for pi, c in enumerate(route):
            loc[c] = (ri, pi)
    return loc


def _route_ok(m, respect_tw: bool) -> bool:
    return m.capacity_ok and (m.time_ok or not respect_tw)


def _relocate_pass(solution: Solution, instance, dist: np.ndarray,
                    neighbours: List[List[int]], respect_tw: bool) -> bool:
    """First-improvement relocation, restricted to each customer's nearest
    neighbours instead of every position in every route."""
    routes = solution.routes
    loc = _locate_customers(routes)
    route_dist = [evaluate_route(r, instance, dist).distance for r in routes]

    for si, src in enumerate(routes):
        for pi, cust in enumerate(list(src)):
            trial_src = src[:pi] + src[pi + 1:]
            m_src_after = evaluate_route(trial_src, instance, dist)
            if not _route_ok(m_src_after, respect_tw):
                continue

            src_before = route_dist[si]
            best_gain = 0.0
            best_plan = None

            tried_positions = set()
            for nb in neighbours[cust]:
                if nb not in loc:
                    continue
                di, nb_pos = loc[nb]
                if di == si and nb_pos == pi:
                    continue
                base = trial_src if di == si else routes[di]
                dest_before = m_src_after.distance if di == si else route_dist[di]
                for pos in (nb_pos, nb_pos + 1):
                    pos = min(pos, len(base))
                    if (di, pos) in tried_positions:
                        continue
                    tried_positions.add((di, pos))

                    trial_dest = base[:pos] + [cust] + base[pos:]
                    m = evaluate_route(trial_dest, instance, dist)
                    if not _route_ok(m, respect_tw):
                        continue

                    if di == si:
                        delta = m.distance - src_before
                    else:
                        delta = (m_src_after.distance + m.distance) - (src_before + dest_before)

                    gain = -delta
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_plan = (di, pos)

            if best_plan is not None:
                di, pos = best_plan
                new_routes = [list(r) for r in routes]
                new_routes[si] = trial_src
                base = new_routes[si] if di == si else new_routes[di]
                base.insert(pos, cust)
                solution.routes = [r for r in new_routes if r]
                return True

    return False


def _swap_pass(solution: Solution, instance, dist: np.ndarray,
               neighbours: List[List[int]], respect_tw: bool) -> bool:
    """First-improvement inter-route swap between nearest-neighbour customers."""
    routes = solution.routes
    loc = _locate_customers(routes)
    route_dist = [evaluate_route(r, instance, dist).distance for r in routes]

    for c1, (i, a) in list(loc.items()):
        r1 = routes[i]
        base1 = route_dist[i]
        for c2 in neighbours[c1]:
            if c2 not in loc:
                continue
            j, b = loc[c2]
            if j == i:
                continue
            r2 = routes[j]
            base2 = route_dist[j]

            t1 = r1[:a] + [c2] + r1[a + 1:]
            t2 = r2[:b] + [c1] + r2[b + 1:]
            m1 = evaluate_route(t1, instance, dist)
            m2 = evaluate_route(t2, instance, dist)
            if not (_route_ok(m1, respect_tw) and _route_ok(m2, respect_tw)):
                continue
            delta = (m1.distance + m2.distance) - (base1 + base2)
            if delta < -1e-9:
                routes[i] = t1
                routes[j] = t2
                return True
    return False


def _two_opt_pass(solution: Solution, instance, dist: np.ndarray,
                   respect_tw: bool) -> bool:
    """First-improvement intra-route 2-opt."""
    for ri, route in enumerate(solution.routes):
        n = len(route)
        if n < 2:
            continue
        base = evaluate_route(route, instance, dist).distance
        for i in range(n - 1):
            for j in range(i + 1, n):
                trial = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                m = evaluate_route(trial, instance, dist)
                if respect_tw and not m.time_ok:
                    continue
                if m.distance - base < -1e-9:
                    solution.routes[ri] = trial
                    return True
    return False


def _local_search(
    solution: Solution,
    instance,
    dist: np.ndarray,
    eliminate_fn,
    neighbours: List[List[int]],
    rng: random.Random,
    max_passes: int,
    respect_tw: bool,
) -> Solution:
    """Repeat relocate/swap/2-opt/elimination until a pass makes no change,
    or max_passes is reached."""
    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        if _relocate_pass(solution, instance, dist, neighbours, respect_tw):
            improved = True
        if _swap_pass(solution, instance, dist, neighbours, respect_tw):
            improved = True
        if _two_opt_pass(solution, instance, dist, respect_tw):
            improved = True

        before = len(solution.active_routes)
        solution = eliminate_fn(solution, rng)
        if len(solution.active_routes) < before:
            improved = True

        passes += 1

    solution.routes = [r for r in solution.routes if r]
    return solution


@register_solver("grasp")
class GRASP(MOSolver):

    DEFAULTS = dict(
        iterations=300,
        alpha=0.3,
        local_search_max_passes=50,
        neighbor_k=12,
        time_limit=None,
        archive_max_size=None,
        tw_mode="hard",
    )

    def __init__(self, instance, objective_names, params=None, *, seed: int = 0):
        # MOSolver defaults tw_mode to "soft" without seeing DEFAULTS above,
        # so fill it in before super().__init__().
        params = dict(params or {})
        params.setdefault("tw_mode", self.DEFAULTS["tw_mode"])
        super().__init__(instance, objective_names, params, seed=seed)

    def _p(self, key: str):
        return self.params.get(key, self.DEFAULTS[key])

    def _solve(self) -> List[Solution]:
        n_iterations = int(self._p("iterations"))
        alpha = float(self._p("alpha"))
        max_passes = int(self._p("local_search_max_passes"))
        neighbor_k = int(self._p("neighbor_k"))
        time_limit = self._p("time_limit")
        archive_max_size = self._p("archive_max_size")

        operators = bind_operators(self.instance, self.dist, respect_tw=self.respect_tw)
        eliminate = operators[3]
        neighbours = _nearest_neighbor_lists(self.instance, self.dist, neighbor_k)

        archive: List[Solution] = []
        sol = None
        start = time.perf_counter()

        for t in range(n_iterations):
            if time_limit is not None and (time.perf_counter() - start) >= float(time_limit):
                break

            sol = _construct(self.instance, self.dist, alpha, self.rng, self.respect_tw)
            sol = _local_search(sol, self.instance, self.dist, eliminate,
                                 neighbours, self.rng, max_passes, self.respect_tw)
            archive = self.update_archive(archive, [sol], max_size=archive_max_size)

            self.log_generation(archive)

        return archive if archive else ([sol] if sol is not None else [])
