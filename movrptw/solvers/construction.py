"""Construction heuristics and neighbourhood operators shared by solvers.

Deliberately solver-agnostic so a GA, Simulated Annealing, Tabu Search or VNS
can all reuse the same building blocks.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from ..instance import Instance
from ..solution import Solution, evaluate_route


def feasible_insert_cost(
    route: List[int],
    pos: int,
    cust: int,
    instance: Instance,
    dist: np.ndarray,
    *,
    respect_tw: bool = True,
) -> Optional[float]:
    """Extra distance of inserting `cust` at `pos`, or None if not allowed.

    Capacity is always enforced. Time windows only when respect_tw=True; in
    soft mode a late insertion is allowed and its lateness accrues into
    total_tardiness instead of being blocked here.
    """
    trial = route[:pos] + [cust] + route[pos:]
    m = evaluate_route(trial, instance, dist)
    if not m.capacity_ok:
        return None
    if respect_tw and not m.time_ok:
        return None
    base = evaluate_route(route, instance, dist).distance
    return m.distance - base


def nearest_neighbour_construct(
    instance: Instance,
    dist: np.ndarray,
    rng: random.Random,
    *,
    randomized: bool = True,
    k: int = 3,
    respect_tw: bool = True,
) -> Solution:
    """Build a solution by greedily extending the current route from among
    the k cheapest feasible next customers (randomized-greedy when
    randomized=True), opening a new vehicle whenever nothing fits."""
    unrouted = list(range(1, instance.size))
    rng.shuffle(unrouted)
    routes: List[List[int]] = []

    current: List[int] = []
    while unrouted:
        candidates = []
        for cust in unrouted:
            cost = feasible_insert_cost(
                current, len(current), cust, instance, dist, respect_tw=respect_tw
            )
            if cost is not None:
                candidates.append((cost, cust))
        if not candidates:
            if current:
                routes.append(current)
                current = []
                continue
            # Nothing fits this customer at all — force it into its own route.
            cust = unrouted.pop(0)
            routes.append([cust])
            continue

        candidates.sort(key=lambda t: t[0])
        pick = candidates[rng.randrange(min(k, len(candidates)))] if randomized else candidates[0]
        chosen = pick[1]
        current.append(chosen)
        unrouted.remove(chosen)

    if current:
        routes.append(current)
    return Solution(routes)


def relocate(solution: Solution, rng: random.Random) -> Solution:
    """Move one customer from one route to a random position in another."""
    s = solution.copy()
    routes = [r for r in s.routes if r]
    if not routes:
        return s
    src = rng.choice(routes)
    cust = src.pop(rng.randrange(len(src)))
    dest_options = [r for r in s.routes if r is not src] + [[]]
    dest = rng.choice(dest_options)
    if not dest and dest not in s.routes:
        s.routes.append(dest)
    dest.insert(rng.randrange(len(dest) + 1), cust)
    s.routes = [r for r in s.routes if r]
    return s


def swap(solution: Solution, rng: random.Random) -> Solution:
    """Swap two customers between (or within) routes."""
    s = solution.copy()
    flat = [(ri, ci) for ri, r in enumerate(s.routes) for ci in range(len(r))]
    if len(flat) < 2:
        return s
    (r1, c1), (r2, c2) = rng.sample(flat, 2)
    s.routes[r1][c1], s.routes[r2][c2] = s.routes[r2][c2], s.routes[r1][c1]
    return s


def two_opt(solution: Solution, rng: random.Random) -> Solution:
    """Reverse a random segment within a single route."""
    s = solution.copy()
    routes = [r for r in s.routes if len(r) >= 2]
    if not routes:
        return s
    r = rng.choice(routes)
    i, j = sorted(rng.sample(range(len(r)), 2))
    r[i : j + 1] = reversed(r[i : j + 1])
    return s


NEIGHBOURHOODS = (relocate, swap, two_opt)


def bind_operators(instance: Instance, dist: np.ndarray, *, respect_tw: bool = True):
    """Return (relocate, swap, two_opt, eliminate) with eliminate closed over
    the instance, since it needs feasibility checks. respect_tw=False (soft
    mode) lets elimination reinsert customers into late-arriving positions,
    trading tardiness for fewer vehicles."""

    def _eliminate(solution: Solution, rng: random.Random) -> Solution:
        active = [r for r in solution.copy().routes if r]
        if len(active) < 2:
            return solution
        target = min(active, key=len)
        others = [r for r in active if r is not target]
        base = Solution([list(r) for r in others])
        rebuilt = greedy_insert(
            base, list(target), instance, dist, rng, respect_tw=respect_tw
        )
        if len(rebuilt.active_routes) < len(active):
            from ..solution import evaluate
            ev = evaluate(rebuilt, instance, dist)
            hard_ok = ev.capacity_violation <= 1e-6 and ev.unserved == 0
            if respect_tw:
                hard_ok = hard_ok and ev.time_window_violation <= 1e-6
            if hard_ok:
                return rebuilt
        return solution

    return (relocate, swap, two_opt, _eliminate)


def random_neighbour(solution: Solution, rng: random.Random, operators=NEIGHBOURHOODS) -> Solution:
    """Apply a uniformly chosen operator from `operators`."""
    op = rng.choice(operators)
    return op(solution, rng)


def greedy_insert(
    solution: Solution,
    customers: List[int],
    instance: Instance,
    dist: np.ndarray,
    rng: random.Random,
    *,
    respect_tw: bool = True,
) -> Solution:
    """Insert each customer at its cheapest feasible position, opening a new
    single-customer route if nothing fits."""
    s = solution.copy()
    order = list(customers)
    rng.shuffle(order)
    for cust in order:
        best_cost = None
        best_pos = None
        for route in s.routes:
            for pos in range(len(route) + 1):
                cost = feasible_insert_cost(
                    route, pos, cust, instance, dist, respect_tw=respect_tw
                )
                if cost is not None and (best_cost is None or cost < best_cost):
                    best_cost, best_pos = cost, (route, pos)
        if best_pos is None:
            s.routes.append([cust])
        else:
            route, pos = best_pos
            route.insert(pos, cust)
    s.routes = [r for r in s.routes if r]
    return s


def route_crossover(
    parent_a: Solution,
    parent_b: Solution,
    instance: Instance,
    dist: np.ndarray,
    rng: random.Random,
    *,
    respect_tw: bool = True,
) -> Solution:
    """Route-based crossover: copy whole routes from parent_a into the child,
    remove those customers from parent_b's routes, then greedily reinsert
    whatever's left unplaced."""
    a_routes = parent_a.active_routes
    if not a_routes:
        return parent_b.copy()
    n_inherit = rng.randint(1, max(1, len(a_routes) // 2))
    inherited = rng.sample(a_routes, min(n_inherit, len(a_routes)))
    inherited_custs = {c for r in inherited for c in r}

    child_routes = [list(r) for r in inherited]
    leftover_routes = [
        [c for c in r if c not in inherited_custs] for r in parent_b.active_routes
    ]
    child_routes += [r for r in leftover_routes if r]
    child = Solution([r for r in child_routes if r])

    placed = set(child.all_customers())
    missing = [c for c in range(1, instance.size) if c not in placed]
    if missing:
        child = greedy_insert(child, missing, instance, dist, rng, respect_tw=respect_tw)
    return child
