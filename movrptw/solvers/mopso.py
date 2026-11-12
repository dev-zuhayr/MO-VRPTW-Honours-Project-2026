"""MOPSO solver adapter for the movrptw framework."""
from __future__ import annotations

import random
from typing import List, Dict

from ..solution import Solution
from .. import tw
from .base import register_solver
from .construction import nearest_neighbour_construct
from .mo_base import MOSolver

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


def _elimination_improves(routes, base_routes, instance, dist, soft_tw):
    """Should a route-elimination result replace what it was tried against?

    Fewer vehicles always wins (that is the point of the operator). On a tie
    in vehicle count, soft mode must also check tardiness: without this,
    elimination gets accepted "for free" whenever it doesn't add a vehicle,
    even if it drags tardiness up by thousands of units for zero benefit.
    """
    v_new  = len([r for r in routes if r])
    v_base = len([r for r in base_routes if r])
    if v_new != v_base:
        return v_new < v_base
    if not soft_tw:
        return True
    tard_new  = sum(_route_tardiness(r, instance, dist) for r in routes)
    tard_base = sum(_route_tardiness(r, instance, dist) for r in base_routes)
    return tard_new <= tard_base + 1e-9


def _copy(routes):
    """Deep-ish copy of a route list, creating new lists with the same customer IDs."""
    return [list(r) for r in routes]


def _simple_eval(routes, dist):
    """Quick (distance, vehicle count) pair, cheaper than a full Solution
    evaluation, used inside or-opt's inner search loop.
    """
    d = round(_sol_dist(routes, dist), 2)
    v = len([r for r in routes if r])
    return (d, v)


def _dominates(a: Dict, b: Dict, obj_names) -> bool:
    """True if solution a beats b: at least as good on everything, and
    strictly better on at least one objective.
    """
    return (all(a[o] <= b[o] for o in obj_names) and
            any(a[o] <  b[o] for o in obj_names))


# ── construction ──────────────────────────────────────────────────────────────

def _nn_construct(instance, dist, rng):
    """Nearest-neighbour construction: builds each route by always adding
    whichever unvisited customer is closest and still feasible. Always
    respects hard time windows, even in soft mode, so particles start
    from a decent, low-tardiness solution.
    """
    n = instance.n_customers
    unvisited = list(range(1, n + 1))
    rng.shuffle(unvisited)
    routes = []
    while unvisited:
        route, load, t, prev = [], 0.0, 0.0, 0
        while unvisited:
            best, best_d = None, float('inf')
            for c in unvisited:
                if load + instance.demands[c] > instance.capacity:
                    continue
                travel = dist[prev, c]
                arrival = max(t + travel, instance.ready[c])
                if arrival > instance.due[c]:
                    continue
                if travel < best_d:
                    best_d, best = travel, c
            if best is None:
                break
            route.append(best)
            load += instance.demands[best]
            t = max(t + dist[prev, best],
                    instance.ready[best]) + instance.service[best]
            prev = best
            unvisited.remove(best)
        if route:
            routes.append(route)
        # any remaining unvisited get their own single-customer routes
        if unvisited and not route:
            c = unvisited.pop(0)
            routes.append([c])
    return routes


def _tw_construct(instance, dist):
    """Greedy construction that visits customers in order of their due
    time (earliest deadline first), starting a new route whenever the
    next customer won't fit in the current one.
    """
    n = instance.n_customers
    order = sorted(range(1, n + 1), key=lambda c: instance.due[c])
    routes, route, load, t, prev = [], [], 0.0, 0.0, 0
    for c in order:
        arrival = max(t + dist[prev, c], instance.ready[c])
        if (load + instance.demands[c] <= instance.capacity and
                arrival <= instance.due[c]):
            route.append(c)
            load += instance.demands[c]
            t = arrival + instance.service[c]
            prev = c
        else:
            if route:
                routes.append(route)
            route = [c]
            load = instance.demands[c]
            t = max(dist[0, c], instance.ready[c]) + instance.service[c]
            prev = c
    if route:
        routes.append(route)
    return routes


# ── velocity ──────────────────────────────────────────────────────────────────

def _compute_velocity(routes_a, routes_b, n_moves, rng):
    """Finds customers who sit in a different spot in routes_b than in
    routes_a, and picks a random n_moves of them to "move toward",
    this list of (customer, target_route, target_position) is the velocity.
    """
    pos_b = {}
    for ri, route in enumerate(routes_b):
        for pi, c in enumerate(route):
            pos_b[c] = (ri, pi)
    pos_a = {}
    for ri, route in enumerate(routes_a):
        for pi, c in enumerate(route):
            pos_a[c] = (ri, pi)
    different = [c for c in pos_b if c in pos_a and pos_a[c] != pos_b[c]]
    rng.shuffle(different)
    return [(c, pos_b[c][0], pos_b[c][1]) for c in different[:n_moves]]


def _best_feasible_insert(routes, customer, target_route_idx, instance, dist,
                           soft_tw=False):
    """Cheapest feasible insertion point for ``customer``, biased toward
    ``target_route_idx``.

    Tries every position in ``routes[target_route_idx]`` first, so the guided
    direction of a relocation (move toward where the customer sits in the
    personal best / leader) is preserved. Only if no position in that route
    is feasible does it fall back to searching every other route. Returns
    ``(route_idx, position)`` of the cheapest feasible slot found, or
    ``None`` if no existing route can hold ``customer`` anywhere.
    """
    def best_in_route(ri):
        route = routes[ri]
        best_pos, best_cost = None, float('inf')
        for pi in range(len(route) + 1):
            route.insert(pi, customer)
            cap_ok = _route_feasible_capacity(route, instance)
            tw_ok  = soft_tw or _route_feasible_tw(route, instance, dist)
            if cap_ok and tw_ok:
                cost = _sol_dist([route], dist)
                if cost < best_cost:
                    best_cost, best_pos = cost, pi
            route.pop(pi)
        return best_pos

    if 0 <= target_route_idx < len(routes):
        pos = best_in_route(target_route_idx)
        if pos is not None:
            return target_route_idx, pos

    best_ri, best_pi, best_cost = None, None, float('inf')
    for ri in range(len(routes)):
        if ri == target_route_idx:
            continue
        route = routes[ri]
        for pi in range(len(route) + 1):
            route.insert(pi, customer)
            cap_ok = _route_feasible_capacity(route, instance)
            tw_ok  = soft_tw or _route_feasible_tw(route, instance, dist)
            if cap_ok and tw_ok:
                cost = _sol_dist([route], dist)
                if cost < best_cost:
                    best_cost, best_ri, best_pi = cost, ri, pi
            route.pop(pi)
    if best_ri is not None:
        return best_ri, best_pi
    return None


def _apply_velocity(routes, velocity, instance, dist, soft_tw=False):
    """Apply a list of (customer, target_route, target_position) relocations.

    Each relocation is applied via best-feasible-insertion-near-target
    (:func:`_best_feasible_insert`) rather than requiring the exact target
    position: that exact position is rarely feasible once the customer is
    moved into a different route context, which is what was causing most
    guided (cognitive/social) moves to be rejected outright. Searching for
    the cheapest feasible slot near the target keeps the guided direction
    while making acceptance far more likely.

    Returns ``(routes, n_accepted, n_attempted)`` so callers can measure the
    guided-move acceptance rate.
    """
    routes = _copy(routes)
    accepted = 0
    attempted = 0
    for c, to_r, _to_p in velocity:
        attempted += 1
        from_r = from_p = None
        for ri, route in enumerate(routes):
            if c in route:
                from_r, from_p = ri, route.index(c)
                break
        if from_r is None:
            continue
        routes[from_r].pop(from_p)
        empty = not routes[from_r]
        if empty:
            routes.pop(from_r)
            adj_to_r = to_r - 1 if to_r > from_r else to_r
        else:
            adj_to_r = to_r

        placement = _best_feasible_insert(
            routes, c, adj_to_r, instance, dist, soft_tw)
        if placement is not None:
            ri, pi = placement
            routes[ri].insert(pi, c)
            accepted += 1
        elif (_route_feasible_capacity([c], instance) and
              (soft_tw or _route_feasible_tw([c], instance, dist))):
            # No existing route has room -- open a fresh one.
            routes.append([c])
            accepted += 1
        else:
            # Nowhere is feasible (e.g. demand alone exceeds capacity):
            # put the customer back where it was.
            if empty:
                routes.insert(from_r, [c])
            elif from_r < len(routes):
                routes[from_r].insert(from_p, c)
            else:
                routes.append([c])

    return [r for r in routes if r], accepted, attempted


# ── local search ──────────────────────────────────────────────────────────────

def _or_opt(routes, instance, dist, n_attempts=3, soft_tw=False):
    """Tries relocating each customer to whatever position (in any route)
    gives the best improvement in vehicle count / distance / tardiness,
    repeating up to n_attempts full passes or until nothing improves.
    """
    routes = _copy(routes)
    for _ in range(n_attempts):
        improved = False
        for ri in range(len(routes)):
            pi = 0
            while pi < len(routes[ri]):
                c = routes[ri][pi]
                routes[ri].pop(pi)
                base = _simple_eval(routes, dist)
                # In soft mode a late reinsertion is still allowed, but the
                # move must be charged for the tardiness it adds, otherwise
                # or-opt chases pure distance/vehicle gain and drags tardiness
                # along for free.
                base_route_tard = (
                    [_route_tardiness(r, instance, dist) for r in routes]
                    if soft_tw else None
                )
                best_delta = 0.0
                best_move = (ri, pi)
                for rj in range(len(routes)):
                    for pj in range(len(routes[rj]) + 1):
                        routes[rj].insert(pj, c)
                        cap_ok = _route_feasible_capacity(routes[rj], instance)
                        tw_ok  = soft_tw or _route_feasible_tw(
                            routes[rj], instance, dist)
                        if cap_ok and tw_ok:
                            new_eval = _simple_eval(routes, dist)
                            delta = ((base[1] - new_eval[1]) * 10000
                                     + (base[0] - new_eval[0]))
                            if soft_tw:
                                new_tard = _route_tardiness(
                                    routes[rj], instance, dist)
                                delta -= (new_tard - base_route_tard[rj])
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (rj, pj)
                        routes[rj].pop(pj)
                bri, bpi = best_move
                routes[bri].insert(bpi, c)
                if best_delta > 1e-6:
                    improved = True
                # Advance only when c stayed in routes[ri]; otherwise routes[ri]
                # shrank and the element now at pi is the next unprocessed one.
                if bri == ri:
                    pi += 1
        routes = [r for r in routes if r]
        if not improved:
            break
    return routes


def _route_elimination(routes, instance, dist, rng, soft_tw=False):
    """Removes the shortest route and reinserts its customers wherever's
    cheapest across the remaining routes — this is how a vehicle actually
    gets taken out of use.
    """
    if len(routes) <= 1:
        return routes
    routes = _copy(routes)
    target = min(range(len(routes)), key=lambda i: len(routes[i]))
    to_insert = routes.pop(target)
    for c in to_insert:
        best_cost, best_ri, best_pi = float('inf'), None, None
        for ri, route in enumerate(routes):
            for pi in range(len(route) + 1):
                route.insert(pi, c)
                cap_ok = _route_feasible_capacity(route, instance)
                tw_ok  = soft_tw or _route_feasible_tw(route, instance, dist)
                if cap_ok and tw_ok:
                    cost = _sol_dist([route], dist)
                    if soft_tw:
                        cost += _route_tardiness(route, instance, dist)
                    if cost < best_cost:
                        best_cost, best_ri, best_pi = cost, ri, pi
                route.pop(pi)
        if best_ri is not None:
            routes[best_ri].insert(best_pi, c)
        else:
            routes.append([c])
    return [r for r in routes if r]


# ── archive ───────────────────────────────────────────────────────────────────

def _crowding(archive, obj_names):
    """Crowding distance for each archive member: roughly, how isolated a
    solution is from its neighbours in objective space. Edge solutions
    (best on some objective) get infinite distance so they're never
    pruned first.
    """
    n = len(archive)
    if n <= 2:
        return [float('inf')] * n
    dists = [0.0] * n
    for obj in obj_names:
        vals = sorted(range(n), key=lambda i: archive[i][0][obj])
        dists[vals[0]] = dists[vals[-1]] = float('inf')
        r = archive[vals[-1]][0][obj] - archive[vals[0]][0][obj]
        if r > 0:
            for i in range(1, n - 1):
                dists[vals[i]] += (
                    archive[vals[i+1]][0][obj] - archive[vals[i-1]][0][obj]
                ) / r
    return dists


def _obj_key(od: Dict, obj_names) -> tuple:
    """Rounds an objective dict down to a hashable tuple, so two solutions
    with near-identical objectives can be spotted as duplicates.
    """
    return tuple(round(od[o], 6) for o in obj_names)


def _update_archive(archive, od: Dict, routes, obj_names, max_size=50):
    """Rounds an objective dict down to a hashable tuple, so two solutions
    with near-identical objectives can be spotted as duplicates.
    """
    for ao, _ in archive:
        if _dominates(ao, od, obj_names):
            return archive
    archive = [(o, _copy(r)) for o, r in archive
               if not _dominates(od, o, obj_names)]
    # Collapse duplicate objective vectors (identical solutions add no
    # diversity but would otherwise pad the archive and skew the crowding
    # distance tournament in _select_leader).
    seen = set()
    deduped = []
    for o, r in archive:
        key = _obj_key(o, obj_names)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((o, r))
    archive = deduped
    if _obj_key(od, obj_names) not in seen:
        archive.append((od, _copy(routes)))
    if len(archive) > max_size:
        cd = _crowding(archive, obj_names)
        keep = sorted(range(len(archive)), key=lambda i: -cd[i])[:max_size]
        archive = [archive[i] for i in keep]
    return archive


def _select_leader(archive, obj_names, rng):
    """Picks a leader for social velocity via crowding-distance binary
    tournament: grab two random archive members, keep whichever sits in
    the less crowded (more isolated) part of the front.
    """
    if len(archive) == 1:
        return archive[0][1]
    cd = _crowding(archive, obj_names)
    i, j = rng.sample(range(len(archive)), 2)
    return archive[i][1] if cd[i] >= cd[j] else archive[j][1]


# ── main solver ───────────────────────────────────────────────────────────────

@register_solver("mopso")
class MOPSOSolver(MOSolver):
    """Multi-Objective Particle Swarm Optimisation: a swarm of route-list
    "particles" nudge each other toward good solutions via cognitive
    (personal best), social (archive leader), and inertia velocity,
    adapted here to work on discrete relocations instead of continuous
    vectors.
    """
    multi_objective = True
    DEFAULTS = dict(
        n_particles=50,
        n_iterations=300,
        max_archive=50,
        tw_mode="hard",
        debug=False,
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
        """Runs the full particle swarm loop for n_iterations and returns
        the non-dominated archive built up along the way.
        """
        instance    = self.instance
        dist        = self.dist
        rng         = random.Random(self.seed)
        n           = instance.n_customers
        n_particles = int(self._p("n_particles"))
        n_iters     = int(self._p("n_iterations"))
        max_archive = int(self._p("max_archive"))
        tw_mode     = self.tw_mode
        soft_tw     = not self.respect_tw
        obj_names   = self.objective_names
        debug       = bool(self._p("debug"))

        def fw_eval(routes):
            ev = self.evaluate(Solution(routes))
            od = {name: ev.objective(name) for name in obj_names}
            return od, ev

        # ── initialise ────────────────────────────────────────────────────
        particles = []
        for construct in [_tw_construct(instance, dist),
                          _nn_construct(instance, dist, rng),
                          _nn_construct(instance, dist, rng)]:
            # first 3 particles get a proper build + polish so the swarm starts
            # from somewhere decent, not pure randomness
            improved = _or_opt(
                construct, instance, dist, n_attempts=10, soft_tw=soft_tw)
            eliminated = _route_elimination(
                improved, instance, dist, rng, soft_tw)
            particles.append(
                eliminated
                if _elimination_improves(eliminated, improved, instance, dist, soft_tw)
                else improved
            )

        while len(particles) < n_particles:
            # everyone else just gets a quick, unpolished nearest-neighbour build
            particles.append(_nn_construct(instance, dist, rng))

        personal_best     = [_copy(p) for p in particles]
        personal_best_obj: List[Dict] = []
        velocities        = [[] for _ in range(n_particles)]
        # Iterations since each particle's personal best last improved,
        # drives both the exploitation step-size shrink and the
        # stagnation re-initialisation below.
        stagnation        = [0] * n_particles

        archive = []
        for i in range(n_particles):
            od, ev = fw_eval(personal_best[i])
            personal_best_obj.append(od)
            # In soft mode admit ALL solutions (tw_admissible allows tardy ones)
            # In hard mode only admit feasible ones
            if tw.admissible(ev, tw_mode):
                archive = _update_archive(archive, od, personal_best[i],
                                          obj_names, max_archive)

        # ── main loop ─────────────────────────────────────────────────────
        for iteration in range(n_iters):
            max_moves = max(1, int(n * (0.4 - 0.3 * iteration / n_iters)))
            w = 0.9 - 0.5 * iteration / n_iters
            # Anneal the blind non-dominated replacement probability from 0.30
            # (early: preserve diversity) down to 0.05 (late: let particles
            # lock onto good solutions instead of thrashing).
            replace_prob = 0.30 * (1.0 - 0.833 * iteration / n_iters)

            iter_accepted  = 0   # guided-relocation acceptance counters
            iter_attempted = 0   # (debug only, but cheap to always track)
            iter_improved  = 0   # particles whose personal best improved

            for i in range(n_particles):
                if stagnation[i] > 30:
                    # this particle's been stuck for too long — throw it away and
                    # rebuild from scratch rather than let it keep wandering
                    seed_rng = random.Random(rng.randrange(2**31))
                    fresh = nearest_neighbour_construct(
                        instance, dist, seed_rng, respect_tw=not soft_tw)
                    fresh_routes = _or_opt(
                        fresh.routes, instance, dist,
                        n_attempts=3, soft_tw=soft_tw)

                    particles[i]  = fresh_routes
                    velocities[i] = []
                    od, ev = fw_eval(fresh_routes)
                    personal_best[i]     = _copy(fresh_routes)
                    personal_best_obj[i] = od
                    stagnation[i] = 0

                    if tw.admissible(ev, tw_mode):
                        archive = _update_archive(
                            archive, od, fresh_routes, obj_names, max_archive)
                    continue

                leader = (_select_leader(archive, obj_names, rng)
                          if archive else personal_best[i])

                # A particle that hasn't improved in a while (but isn't yet
                # stagnant enough to be re-initialised) gets a much smaller
                # step: exploit the neighbourhood of its current best rather
                # than keep taking large guided jumps.
                particle_max_moves = (
                    max(1, int(n * 0.05)) if stagnation[i] >= 20 else max_moves
                )

                v_cog = _compute_velocity(
                    particles[i], personal_best[i],
                    int(particle_max_moves * rng.random()), rng)
                v_soc = _compute_velocity(
                    particles[i], leader,
                    int(particle_max_moves * rng.random()), rng)

                new_routes, acc1, att1 = _apply_velocity(
                    particles[i], v_cog, instance, dist, soft_tw)
                new_routes, acc2, att2 = _apply_velocity(
                    new_routes, v_soc, instance, dist, soft_tw)
                iter_accepted  += acc1 + acc2
                iter_attempted += att1 + att2

                if velocities[i] and rng.random() < w:
                    n_carry = max(1, int(len(velocities[i]) * w))
                    carried = rng.sample(
                        velocities[i], min(n_carry, len(velocities[i])))
                    new_routes, acc3, att3 = _apply_velocity(
                        new_routes, carried, instance, dist, soft_tw)
                    iter_accepted  += acc3
                    iter_attempted += att3

                if rng.random() < 0.15:
                    cand = _route_elimination(
                        new_routes, instance, dist, rng, soft_tw)
                    if _elimination_improves(cand, new_routes, instance, dist, soft_tw):
                        new_routes = cand

                if rng.random() < 0.05:
                    new_routes = _or_opt(
                        new_routes, instance, dist,
                        n_attempts=2, soft_tw=soft_tw)

                od, ev = fw_eval(new_routes)
                velocities[i] = _compute_velocity(
                    particles[i], new_routes, particle_max_moves, rng)
                particles[i] = new_routes

                # Only update archive with admissible solutions
                if tw.admissible(ev, tw_mode):
                    archive = _update_archive(
                        archive, od, new_routes, obj_names, max_archive)

                if _dominates(od, personal_best_obj[i], obj_names):
                    personal_best[i]     = _copy(new_routes)
                    personal_best_obj[i] = od
                    stagnation[i] = 0
                    iter_improved += 1
                else:
                    stagnation[i] += 1
                    if not _dominates(personal_best_obj[i], od, obj_names):
                        if rng.random() < replace_prob:
                            personal_best[i]     = _copy(new_routes)
                            personal_best_obj[i] = od

            self._iterations += 1
            if archive:
                self._convergence.append(
                    min(o.get("total_distance", 0.0) for o, _ in archive))

            if debug and (iteration % 25 == 0 or iteration == n_iters - 1):
                accept_rate = (100.0 * iter_accepted / iter_attempted
                               if iter_attempted else 0.0)
                avg_accepted = iter_accepted / n_particles
                improved_pct = 100.0 * iter_improved / n_particles
                unique_vecs = len({_obj_key(o, obj_names) for o, _ in archive})
                best_dist = min(
                    (o["total_distance"] for o, _ in archive), default=float("nan"))
                print(
                    f"[mopso debug] iter={iteration:4d} "
                    f"avg_accepted/particle={avg_accepted:5.2f} "
                    f"accept_rate={accept_rate:5.1f}% "
                    f"pb_improved={improved_pct:5.1f}% "
                    f"archive_size={len(archive):3d} "
                    f"unique_vectors={unique_vecs:3d} "
                    f"best_dist={best_dist:.2f}"
                )

        # ── post-process ──────────────────────────────────────────────────
        final_archive = []
        for od, routes in archive:
            better = _or_opt(routes, instance, dist,
                             n_attempts=10, soft_tw=soft_tw)
            elim   = _route_elimination(better, instance, dist, rng, soft_tw)
            if _elimination_improves(elim, better, instance, dist, soft_tw):
                better = elim
            new_od, new_ev = fw_eval(better)
            if tw.admissible(new_ev, tw_mode):
                final_archive = _update_archive(
                    final_archive, new_od, better, obj_names, max_archive)

        # Fall back to main-loop archive if post-processing discards everything.
        if not final_archive:
            final_archive = archive

        return [Solution(routes) for _, routes in final_archive]
