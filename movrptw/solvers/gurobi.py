"""Gurobi epsilon-constraint solver adapter for the movrptw framework."""
from __future__ import annotations

from typing import List

import numpy as np

from ..instance import Instance
from ..solution import Solution
from .base import MetaheuristicSolver, register_solver

# BKS tables for these two sizes were computed with distances rounded
# down to 1 decimal per edge. Solving with the same rounding keeps our
# optimum comparable to those published values, instead of a more
# precise number they never actually reported.
_FLOOR1_SIZES = (25, 50)

# how many vehicle-count steps in a row can fail to improve before we
# give up on the sweep. Used to be 3, but that was too quick to quit,
# two levels landing within rounding noise of each other could trigger
# a stop before a genuinely better level further out.
_SWEEP_PATIENCE = 6

# maps Gurobi's numeric status codes to short, readable names for the
# sweep trace. anything we don't recognise still gets recorded as
# "status_<code>" instead of just disappearing
_STATUS_NAMES = {
    2: "optimal", 9: "time_limit", 3: "infeasible", 4: "inf_or_unbounded",
    5: "unbounded", 11: "interrupted", 13: "suboptimal",
}


def _status_name(code: int) -> str:
    """Return a short name for a Gurobi status code, or a generic fallback if unknown."""
    return _STATUS_NAMES.get(int(code), f"status_{int(code)}")


def _mip_gap(model) -> "float | None":
    """How far off from proven-optimal this solve is (0.0 means proven
    optimal). Returns None if there's no feasible solution yet.

    Gurobi still reports this even when it hits the time limit -- so it
    tells us how close a timed-out solve actually got, not just that it
    ran out of time.
    """
    try:
        if model.SolCount > 0:
            return float(model.MIPGap)
    except Exception:
        pass
    return None


def _floor1_matrix(instance) -> np.ndarray:
    """Straight-line distance between every pair of customers, rounded down
    to 1 decimal place.

    We can't just call instance.distance_matrix(rounding="floor1") here --
    that method caches its result the first time it's called, and by the
    time GurobiSolver runs, another solver has usually already triggered
    that cache with full-precision distances. Calling it again would just
    hand back the cached (wrong) version. Building it ourselves avoids that,
    and avoids touching the shared instance other solvers are also using.
    """
    c = np.asarray(instance.coords, dtype=float)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    return np.floor(d * 10) / 10


def _build(instance, dist, num_vehicles, epsilon=None, threads=1):
    """Build a Gurobi model for the MOVRPTW instance with the given parameters."""
    import gurobipy as gp
    from gurobipy import GRB

    n = instance.size
    K = range(num_vehicles)
    N = range(n)
    M = float(max(instance.due) + max(instance.service))

    m = gp.Model("MOVRPTW")
    m.setParam('OutputFlag', 0)
    # Gurobi tries to grab every core on the machine by default. If we're
    # running several instances in parallel (one per joblib worker), each
    # one doing that at once would massively oversubscribe the machine,
    # so we cap threads per solve, and get parallelism from running
    # multiple instances side by side instead.
    m.setParam('Threads', threads)

    x = m.addVars(n, n, num_vehicles, vtype=GRB.BINARY, name="x")
    t = m.addVars(n, num_vehicles, vtype=GRB.CONTINUOUS, name="t")
    u = m.addVars(num_vehicles, vtype=GRB.BINARY, name="u")

    for i in range(1, n):
        m.addConstr(gp.quicksum(x[i, j, k] for j in N if j != i for k in K) == 1)
    for k in K:
        for i in N:
            m.addConstr(
                gp.quicksum(x[i, j, k] for j in N if j != i) ==
                gp.quicksum(x[j, i, k] for j in N if j != i)
            )
        m.addConstr(gp.quicksum(x[0, j, k] for j in range(1, n)) == u[k])
        m.addConstr(gp.quicksum(x[i, 0, k] for i in range(1, n)) == u[k])
        m.addConstr(gp.quicksum(
            instance.demands[i] * x[i, j, k]
            for i in range(1, n) for j in N if j != i
        ) <= instance.capacity)
        for i in N:
            for j in range(1, n):
                if i != j:
                    m.addConstr(
                        t[j, k] >= t[i, k] + instance.service[i]
                        + dist[i, j] - M * (1 - x[i, j, k])
                    )
            for i2 in N:
                m.addConstr(t[i2, k] >= instance.ready[i2])
                m.addConstr(t[i2, k] <= instance.due[i2])

    total_dist = gp.quicksum(dist[i, j] * x[i, j, k]
                             for i in N for j in N if i != j for k in K)
    num_used = gp.quicksum(u[k] for k in K)

    if epsilon is not None:
        m.addConstr(num_used <= epsilon, name="eps")
        m.setObjective(total_dist, GRB.MINIMIZE)
    else:
        m.setObjective(num_used, GRB.MINIMIZE)

    return m, x, u, total_dist, num_used


def _extract_routes(x_vars, n, num_vehicles):
    """Reconstruct each vehicle's visiting order from the solved x[i,j,k]
    binary variables by walking each route from the depot (node 0)
    until no further customer is reachable.
    """
    routes = []
    for k in range(num_vehicles):
        route, current, visited = [], 0, set()
        for _ in range(n):
            nxt = next((j for j in range(1, n)
                        if j not in visited and x_vars[current, j, k].X > 0.5), None)
            if nxt is None:
                break
            route.append(nxt)
            visited.add(nxt)
            current = nxt
        if route:
            routes.append(route)
    return routes


@register_solver("gurobi")
class GurobiSolver(MetaheuristicSolver):
    """Exact solver via a two-step epsilon-constraint method: first find
    the minimum feasible vehicle count, then sweep vehicle-count caps
    upward from there, minimising distance at each level, to trace out
    the Pareto front. See _solve() and module docstring for details.
    """
    multi_objective = True
    DEFAULTS = dict(time_limit=300, threads=1)

    def __init__(self, instance: Instance, objective_names, params=None, *, seed: int = 0):
        super().__init__(instance, objective_names, params, seed=seed)
        # For 25/50 customers, switch to the same rounded distances the
        # published BKS tables use, so what we find is actually comparable
        # to them. At 100 customers we just keep the normal full-precision
        # distances, Gurobi doesn't run at that size anyway, so there's
        # no BKS table to match.
        if instance.n_customers in _FLOOR1_SIZES:
            self.dist = _floor1_matrix(instance)

    def _p(self, key):
        return self.params.get(key, self.DEFAULTS[key])

    def _solve(self) -> List[Solution]:
        instance = self.instance
        dist = self.dist
        num_vehicles = instance.num_vehicles
        time_limit = float(self._p("time_limit"))
        threads = int(self._p("threads"))

        # keeps a record of every solve attempted, the initial veh_min
        # step plus every vehicle count tried in the sweep, so we can
        # tell later which ones were actually proven optimal and which
        # just timed out, not only the one we end up calling "best".
        self.sweep_trace: List[dict] = []

        def _trace(phase, epsilon, model, *, distance=None, num_veh=None,
                   feasible=None, in_pool=False):
            """Append one record to self.sweep_trace for this solve attempt."""
            self.sweep_trace.append({
                "phase": phase,
                "epsilon": epsilon,
                "status": int(model.Status),
                "status_name": _status_name(model.Status),
                "mip_gap": _mip_gap(model),
                "sol_count": int(model.SolCount),
                "runtime_s": round(float(model.Runtime), 2),
                "distance": distance,
                "num_vehicles": num_veh,
                "feasible": feasible,
                "in_pool": in_pool,
            })

        # Step 1: find veh_min
        m, x, u, td, nu = _build(instance, dist, num_vehicles, epsilon=None, threads=threads)
        m.setParam('TimeLimit', time_limit)
        m.optimize()
        veh_min_val = (int(round(nu.getValue()))
                       if m.Status in (2, 9) and m.SolCount > 0 else None)
        _trace("veh_min", None, m, num_veh=veh_min_val)
        if veh_min_val is None:
            return [Solution([])]
        veh_min = veh_min_val

        # Step 2: sweep
        pool = []
        prev_dist = float('inf')
        no_improve = 0

        for v in range(veh_min, num_vehicles + 1):
            m_v, x_v, u_v, td_v, nu_v = _build(instance, dist, num_vehicles, epsilon=v, threads=threads)
            m_v.setParam('TimeLimit', time_limit)
            m_v.optimize()

            if m_v.Status not in (2, 9) or m_v.SolCount == 0:
                _trace("sweep", v, m_v)
                continue

            # we compare the real distance here, not the rounded-to-2-decimals
            # version used for reporting, otherwise two levels that are
            # actually slightly different could round to the same number and
            # look like "no improvement" when one was genuinely a bit better.
            d_true = float(td_v.getValue())
            d = round(d_true, 2)
            routes = _extract_routes(x_v, instance.size, num_vehicles)
            sol = Solution(routes)
            ev = self.evaluate(sol)
            if ev.feasible:
                # stamp this solution with which vehicle-count level it
                # came from, and whether that level was proven optimal or
                # just timed out, so whatever picks the "best" solution
                # later can trace it back
                sol.gurobi_epsilon = v
                sol.gurobi_status = int(m_v.Status)
                sol.gurobi_status_name = _status_name(m_v.Status)
                sol.gurobi_mip_gap = _mip_gap(m_v)
                pool.append(sol)
            _trace("sweep", v, m_v, distance=d_true,
                   num_veh=int(ev.num_vehicles), feasible=bool(ev.feasible),
                   in_pool=bool(ev.feasible))

            if d_true >= prev_dist:
                no_improve += 1
                if no_improve >= _SWEEP_PATIENCE:
                    break
            else:
                no_improve = 0
                prev_dist = d_true

            self._iterations += 1
            self._convergence.append(d)

        return pool if pool else [Solution([])]
