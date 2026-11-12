"""ALNS solver adapter for the movrptw framework."""
from __future__ import annotations

import random
import math
from typing import List

from ..solution import Solution
from .. import tw
from .base import register_solver, _objective_scales
from .construction import nearest_neighbour_construct
from .mo_base import MOSolver

def _route_feasible_capacity(route, instance):
    """True if the total demand on this route fits in one vehicle."""
    load = float(sum(instance.demands[c] for c in route))
    return load <= instance.capacity


def _route_feasible_tw(route, instance, dist):
    """Walks the route customer by customer, tracking the clock.
    Returns False the moment someone gets served after their due time.
    """
    t, prev = 0.0, 0
    for c in route:
        t += dist[prev, c]
        t = max(t, instance.ready[c])
        if t > instance.due[c]:
            return False
        t += instance.service[c]
        prev = c
    return True


def _route_tardiness(route, instance, dist):
    """Same walk as _route_feasible_tw, but instead of failing on a late
    arrival it just adds up how late everyone was.
    """
    t, prev, tardiness = 0.0, 0, 0.0
    for c in route:
        t += dist[prev, c]
        t = max(t, instance.ready[c])
        tardiness += max(0.0, t - instance.due[c])
        t += instance.service[c]
        prev = c
    return tardiness


def _route_feasible(route, instance, dist):
    """True if the route respects both capacity and time windows."""
    return (_route_feasible_capacity(route, instance) and
            _route_feasible_tw(route, instance, dist))


def _sol_dist(routes, dist):
    """Total distance of a full solution, depot to depot, across every route."""
    total = 0.0
    for r in routes:
        if not r:
            continue
        total += dist[0, r[0]]
        for i in range(len(r) - 1):
            total += dist[r[i], r[i+1]]
        total += dist[r[-1], 0]
    return total

# ── destroy operators ────────────────────────────────────────────────────────

def _random_removal(routes, n_remove, rng):
    """Simplest destroy: grab random customers off whatever route they're on."""    
    routes = [list(r) for r in routes]
    all_c = [c for r in routes for c in r]
    removed = rng.sample(all_c, min(n_remove, len(all_c)))
    for c in removed:
        for r in routes:
            if c in r:
                r.remove(c)
                break
    return [r for r in routes if r], removed


def _worst_removal(routes, n_remove, dist, rng):
    """Removes whoever's costing the most right now, i.e. whoever, if
    pulled out, would save the most distance (prev->c->next collapsing
    down to prev->next).
    """
    routes = [list(r) for r in routes]
    removed = []
    for _ in range(n_remove):
        best_saving, best_c, best_ri = -1, None, None
        for ri, route in enumerate(routes):
            for ci, c in enumerate(route):
                prev = route[ci-1] if ci > 0 else 0
                nxt = route[ci+1] if ci < len(route)-1 else 0
                saving = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                if saving > best_saving:
                    best_saving, best_c, best_ri = saving, c, ri
        if best_c is None:
            break
        routes[best_ri].remove(best_c)
        removed.append(best_c)
    return [r for r in routes if r], removed


def _related_removal(routes, n_remove, dist, rng):
    """Picks one random customer, then removes their geographic neighbours
    too — a cluster of related customers is easier for repair to place
    well together than a scattered random set.
    """
    all_c = [c for r in routes for c in r]
    if not all_c:
        return [list(r) for r in routes], []
    seed_c = rng.choice(all_c)
    related = sorted([c for c in all_c if c != seed_c],
                     key=lambda c: dist[seed_c, c])
    to_remove = [seed_c] + related[:n_remove-1]
    routes = [list(r) for r in routes]
    removed = []
    for c in to_remove:
        for r in routes:
            if c in r:
                r.remove(c)
                removed.append(c)
                break
    return [r for r in routes if r], removed


def _route_elimination(routes, rng):
    """Removes the whole shortest route and throws all its customers back
    into the pool — this is how a vehicle actually gets taken out of use.
    """
    if not routes:
        return [], []
    routes = [list(r) for r in routes]
    target = min(range(len(routes)), key=lambda i: len(routes[i]))
    removed = routes.pop(target)
    return routes, removed


# ── repair operators ─────────────────────────────────────────────────────────

def _best_insert(route, c, instance, dist, soft_tw=False):
    """Tries inserting customer c at a spread of positions along the route
    (not literally every slot, every 10th, to keep this fast) and
    returns whichever spot is cheapest.

    In soft mode a late insertion isn't rejected outright, it's just
    charged for the extra tardiness it introduces, so repair doesn't
    chase pure distance and drag tardiness along for free.
    """
    best_cost, best_pos = float('inf'), None
    step = max(1, len(route) // 10)
    positions = list(range(0, len(route) + 1, step))
    if len(route) not in positions:
        positions.append(len(route))

    base_tardiness = _route_tardiness(route, instance, dist) if soft_tw else 0.0

    for pos in positions:
        route.insert(pos, c)
        cap_ok = _route_feasible_capacity(route, instance)
        tw_ok  = soft_tw or _route_feasible_tw(route, instance, dist)
        if cap_ok and tw_ok:
            prev = route[pos-1] if pos > 0 else 0
            nxt  = route[pos+1] if pos < len(route)-1 else 0
            cost = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
            if soft_tw:
                cost += _route_tardiness(route, instance, dist) - base_tardiness
            if cost < best_cost:
                best_cost, best_pos = cost, pos
        route.pop(pos)
    return best_pos, best_cost


def _greedy_insert(routes, removed, instance, dist, rng, soft_tw=False):
    """For each removed customer, just drop them wherever's cheapest right now."""
    routes = [list(r) for r in routes]
    route_loads = [sum(instance.demands[c] for c in r) for r in routes]
    for c in removed:
        demand = instance.demands[c]
        best_cost, best_ri, best_pos = float('inf'), None, None
        for ri, route in enumerate(routes):
            if route_loads[ri] + demand > instance.capacity:
                continue
            pos, cost = _best_insert(route, c, instance, dist, soft_tw)
            if pos is not None and cost < best_cost:
                best_cost, best_ri, best_pos = cost, ri, pos
        if best_ri is not None:
            routes[best_ri].insert(best_pos, c)
            route_loads[best_ri] += demand
        else:
            routes.append([c])
            route_loads.append(demand)
    return routes


def _regret_insert(routes, removed, instance, dist, rng, soft_tw=False, k=2):
    """Smarter than greedy: for each customer, looks at how much worse
    their 2nd-best spot is compared to their best spot (the "regret").
    Inserts whoever has the biggest regret first, since they're the ones
    who'll get hurt most if we wait and someone else takes their good spot.
    """
    routes = [list(r) for r in routes]
    uninserted = list(removed)
    while uninserted:
        best_regret, best_c, best_ri, best_pos = -float('inf'), None, None, None
        for c in uninserted:
            costs = []
            for ri, route in enumerate(routes):
                pos, cost = _best_insert(route, c, instance, dist, soft_tw)
                if pos is not None:
                    costs.append((cost, ri, pos))
            costs.sort(key=lambda x: x[0])
            if not costs:
                regret, ri_b, pos_b = float('inf'), None, None
            elif len(costs) == 1:
                regret, ri_b, pos_b = costs[0][0], costs[0][1], costs[0][2]
            else:
                regret = sum(costs[i][0] - costs[0][0]
                             for i in range(1, min(k, len(costs))))
                ri_b, pos_b = costs[0][1], costs[0][2]
            if regret > best_regret:
                best_regret, best_c, best_ri, best_pos = regret, c, ri_b, pos_b
        if best_c is None:
            break
        uninserted.remove(best_c)
        if best_ri is not None:
            routes[best_ri].insert(best_pos, best_c)
        else:
            routes.append([best_c])
    return routes


# ── main solver ──────────────────────────────────────────────────────────────

@register_solver("alns")
class ALNSSolver(MOSolver):
    """Adaptive Large Neighbourhood Search: repeatedly tears a solution
    apart (destroy) and rebuilds it (repair), keeping track of which
    destroy/repair pairing tends to work best and leaning on it more
    over time.
    """
    multi_objective = True
    DEFAULTS = dict(
        n_iterations=5000,
        n_remove_frac=0.25,
        start_temp=100.0,
        cooling=0.9995,
        segment_size=100,
        tw_mode="hard",
    )

    def __init__(self, instance, objective_names, params=None, *, seed: int = 0):
        # MOSolver defaults tw_mode to "soft" without seeing DEFAULTS above,
        # so fill it in before super().__init__().
        params = dict(params or {})
        params.setdefault("tw_mode", self.DEFAULTS["tw_mode"])
        super().__init__(instance, objective_names, params, seed=seed)

    def _p(self, key):
        """Look up a parameter, falling back to DEFAULTS if not set."""
        return self.params.get(key, self.DEFAULTS[key])

    def _solve(self) -> List[Solution]:
        """Runs the full destroy-repair loop for n_iterations and returns
        the non-dominated archive built up along the way.
        """
        instance  = self.instance
        dist      = self.dist
        tw_mode   = self.tw_mode
        soft_tw   = not self.respect_tw
        rng_py    = random.Random(self.seed)

        n_remove  = max(1, int(instance.n_customers * float(self._p("n_remove_frac"))))
        n_iters   = int(self._p("n_iterations"))
        temp      = float(self._p("start_temp"))
        cooling   = float(self._p("cooling"))
        seg_size  = int(self._p("segment_size"))
        obj_names = self.objective_names
        obj_scale = _objective_scales(instance)

        def obj_dict(ev):
            """Pulls out this evaluation's objective values as a plain dict."""
            return {n: ev.objective(n) for n in obj_names}

        def dom(a, b):
            """Same rule as the top-level _dominates, just closed over obj_names."""
            return (all(a[o] <= b[o] for o in obj_names) and
                    any(a[o] <  b[o] for o in obj_names))

        def update_arc(archive, od, routes):
            """Local version of _update_archive, see note on that function above."""
            for ao, _ in archive:
                if dom(ao, od):
                    return archive
            archive = [(o, [list(r) for r in rs])
                       for o, rs in archive if not dom(od, o)]
            archive.append((od, [list(r) for r in routes]))
            return archive

        # Initial solution: always start from a hard-TW-respecting construction
        # (mirrors MOPSO's "always uses hard TW for good starts"), even in soft
        # mode, so the search begins from a low-tardiness point instead of
        # whatever nearest-neighbour produces with time windows ignored.
        init    = nearest_neighbour_construct(instance, dist, rng_py,
                                              respect_tw=True)
        current = [list(r) for r in init.active_routes]
        best    = [list(r) for r in current]
        archive = []

        init_ev  = self.evaluate(Solution(current))
        init_od  = obj_dict(init_ev)
        archive  = update_arc(archive, init_od, current)

        DESTROY = ["random", "worst", "related", "route_elim"]
        REPAIR  = ["greedy", "regret"]
        d_w = [1.0]*4; r_w = [1.0]*2
        d_s = [0.0]*4; r_s = [0.0]*2
        d_c = [0]*4;   r_c = [0]*2
        DECAY  = 0.8
        SCORES = {"new_best": 5, "better": 3, "accepted": 1, "rejected": 0}

        curr_ev = init_ev
        curr_od = init_od
        best_ev = init_ev
        best_od = init_od

        pool = []

        for iteration in range(n_iters):
            # pick a destroy and repair operator, weighted by how well
            # they've been performing (roulette wheel: better operators
            # get picked more often)
            di = rng_py.choices(range(4), weights=d_w)[0]
            ri = rng_py.choices(range(2), weights=r_w)[0]

            if DESTROY[di] == "random":
                new_r, removed = _random_removal(current, n_remove, rng_py)
            elif DESTROY[di] == "worst":
                new_r, removed = _worst_removal(current, n_remove, dist, rng_py)
            elif DESTROY[di] == "related":
                new_r, removed = _related_removal(current, n_remove, dist, rng_py)
            else:
                new_r, removed = _route_elimination(current, rng_py)

            if REPAIR[ri] == "greedy":
                new_r = _greedy_insert(new_r, removed, instance, dist,
                                       rng_py, soft_tw)
            else:
                new_r = _regret_insert(new_r, removed, instance, dist,
                                       rng_py, soft_tw)

            cand_sol = Solution(new_r)
            cand_ev  = self.evaluate(cand_sol)

            if not tw.admissible(cand_ev, tw_mode):
                score = "rejected"
            else:
                cand_od = obj_dict(cand_ev)
                archive = update_arc(archive, cand_od, new_r)
                pool.append(cand_sol)

                if dom(cand_od, best_od):
                    best    = [list(r) for r in new_r]
                    best_ev = cand_ev
                    best_od = cand_od
                    current = [list(r) for r in new_r]
                    curr_ev = cand_ev
                    curr_od = cand_od
                    score   = "new_best" # beats our best solution ever
                elif dom(cand_od, curr_od):
                    current = [list(r) for r in new_r]
                    curr_ev = cand_ev
                    curr_od = cand_od
                    score   = "better" # beats where we currently are
                else:
                    # Normalise by a fixed instance-derived scale, not by the
                    # current objective value itself: self-relative
                    # normalisation makes an already-large tardiness
                    # increasingly cheap to grow further, letting it run away
                    # unchecked over the course of the search.
                    delta = sum(
                        (cand_od[o] - curr_od[o]) / obj_scale.get(o, 1.0)
                        for o in obj_names
                    ) / len(obj_names)
                    if delta < 0 or rng_py.random() < math.exp(
                            -delta / max(temp, 1e-10)):
                        current = [list(r) for r in new_r]
                        curr_ev = cand_ev
                        curr_od = cand_od
                        score   = "accepted"
                    else:
                        score = "rejected"

            d_s[di] += SCORES[score]; d_c[di] += 1
            r_s[ri] += SCORES[score]; r_c[ri] += 1

            if (iteration + 1) % seg_size == 0:
                # every seg_size iterations, update how much we trust each
                # operator based on its average score this segment, then
                # reset the counters
                for i in range(4):
                    if d_c[i] > 0:
                        d_w[i] = DECAY*d_w[i] + (1-DECAY)*d_s[i]/d_c[i]
                    d_s[i] = d_c[i] = 0
                for i in range(2):
                    if r_c[i] > 0:
                        r_w[i] = DECAY*r_w[i] + (1-DECAY)*r_s[i]/r_c[i]
                    r_s[i] = r_c[i] = 0

            temp *= cooling
            self._iterations += 1
            self._convergence.append(best_od.get("total_distance", 0.0))

        archive_sols = [Solution(routes) for _, routes in archive]
        return archive_sols + [Solution(best)] if archive_sols else [Solution(best)]
