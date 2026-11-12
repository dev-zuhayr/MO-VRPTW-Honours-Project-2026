"""Solution representation and evaluation for VRPTW.

A :class:`Solution` is an ordered set of *routes*. Each route is a list of
customer indices (the depot at index 0 is implicit at the start and end).

Route timing follows the standard VRPTW rules:

* The vehicle leaves the depot at time ``depot.ready_time`` (usually 0).
* Travelling ``i -> j`` takes ``dist[i, j]`` time units.
* If the vehicle arrives at a customer before its ``ready_time`` it *waits*.
* Service begins at ``max(arrival, ready_time)`` and lasts ``service_time``.
* *Tardiness* at a node is ``max(0, arrival - due_time)``.

Time windows are treated as **hard** for feasibility by default (arrival must
not exceed ``due_time`` and the vehicle must return to the depot by the
horizon), while tardiness is still computed so it can be used as a soft
objective if desired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .instance import Instance


@dataclass
class RouteMetrics:
    """Per-route evaluation results."""

    load: float
    distance: float
    wait: float
    tardiness: float
    capacity_ok: bool
    time_ok: bool
    finish_time: float  # depot return time


@dataclass
class Evaluation:
    """Full evaluation of a :class:`Solution` against an instance."""

    num_vehicles: int
    total_distance: float
    total_wait: float
    total_tardiness: float
    route_imbalance: float          # max - min route distance
    makespan: float                 # latest depot return time
    feasible: bool
    capacity_violation: float       # total demand over capacity
    time_window_violation: float    # total tardiness (hard TW)
    unserved: int                   # missing + duplicated customers
    per_route: List[RouteMetrics] = field(default_factory=list)

    def objective(self, name: str) -> float:
        """Return a named objective value (all are minimised)."""
        try:
            return float(getattr(self, _OBJECTIVE_ATTR[name]))
        except KeyError as exc:  # pragma: no cover - guarded by config validation
            raise KeyError(
                f"Unknown objective '{name}'. Known: {sorted(_OBJECTIVE_ATTR)}"
            ) from exc

    def vector(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.objective(n) for n in names], dtype=float)


# Objective name -> Evaluation attribute. Add entries here for new objectives.
_OBJECTIVE_ATTR: Dict[str, str] = {
    "num_vehicles": "num_vehicles",
    "total_distance": "total_distance",
    "total_wait": "total_wait",
    "total_tardiness": "total_tardiness",
    "route_imbalance": "route_imbalance",
    "makespan": "makespan",
}

AVAILABLE_OBJECTIVES = tuple(_OBJECTIVE_ATTR)


@dataclass
class Solution:
    """A candidate solution: a list of routes over customer indices."""

    routes: List[List[int]]

    def copy(self) -> "Solution":
        return Solution([list(r) for r in self.routes])

    @property
    def active_routes(self) -> List[List[int]]:
        return [r for r in self.routes if r]

    def all_customers(self) -> List[int]:
        return [c for r in self.routes for c in r]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Solution(vehicles={len(self.active_routes)}, customers={len(self.all_customers())})"

def evaluate_route(
    route: Sequence[int],
    instance: Instance,
    dist: np.ndarray,
) -> RouteMetrics:
    """Evaluate a single route (list of customer indices, depot implicit)."""
    if not route:
        return RouteMetrics(0.0, 0.0, 0.0, 0.0, True, True, 0.0)

    demand = instance.demands
    ready = instance.ready
    due = instance.due
    service = instance.service

    load = float(demand[list(route)].sum())
    capacity_ok = load <= instance.capacity + 1e-9

    t = instance.depot.ready_time
    prev = 0  # depot
    total_dist = 0.0
    wait = 0.0
    tardiness = 0.0
    time_ok = True

    for cust in route:
        travel = dist[prev, cust]
        total_dist += travel
        arrival = t + travel
        if arrival < ready[cust]:
            wait += ready[cust] - arrival
            start = ready[cust]
        else:
            start = arrival
        late = max(0.0, arrival - due[cust])
        tardiness += late
        if late > 1e-9:
            time_ok = False
        t = start + service[cust]
        prev = cust

    # Return to the depot.
    total_dist += dist[prev, 0]
    finish = t + dist[prev, 0]
    if finish > instance.horizon + 1e-9:
        time_ok = False
        tardiness += finish - instance.horizon

    return RouteMetrics(
        load=load,
        distance=total_dist,
        wait=wait,
        tardiness=tardiness,
        capacity_ok=capacity_ok,
        time_ok=time_ok,
        finish_time=finish,
    )


def evaluate(solution: Solution, instance: Instance, dist: np.ndarray) -> Evaluation:
    """Evaluate a full solution, computing every supported objective."""
    per_route = [evaluate_route(r, instance, dist) for r in solution.active_routes]

    distances = [m.distance for m in per_route] or [0.0]
    total_distance = float(sum(distances))
    total_wait = float(sum(m.wait for m in per_route))
    total_tardiness = float(sum(m.tardiness for m in per_route))
    makespan = max((m.finish_time for m in per_route), default=0.0)
    route_imbalance = float(max(distances) - min(distances)) if len(distances) > 1 else 0.0

    capacity_violation = float(
        sum(max(0.0, m.load - instance.capacity) for m in per_route)
    )
    tw_violation = float(sum(m.tardiness for m in per_route))

    # Every customer must be served exactly once.
    served = solution.all_customers()
    expected = set(range(1, instance.size))
    unserved = len(expected - set(served)) + (len(served) - len(set(served)))

    feasible = (
        capacity_violation <= 1e-6
        and tw_violation <= 1e-6
        and unserved == 0
        and all(m.capacity_ok and m.time_ok for m in per_route)
    )

    return Evaluation(
        num_vehicles=len(per_route),
        total_distance=total_distance,
        total_wait=total_wait,
        total_tardiness=total_tardiness,
        route_imbalance=route_imbalance,
        makespan=makespan,
        feasible=feasible,
        capacity_violation=capacity_violation,
        time_window_violation=tw_violation,
        unserved=unserved,
        per_route=per_route,
    )
