"""MOACO — Multi-Objective Ant Colony Optimisation"""

from __future__ import annotations

import random
from typing import List, Sequence, Any

import numpy as np

from ..instance import Instance
from ..solution import Solution, evaluate_route
from .base import register_solver
from .mo_base import MOSolver


@register_solver("moaco")
class MOACO(MOSolver):
    """Pareto-archive ant colony optimisation with per-objective pheromones."""

    DEFAULTS = dict(
        n_ants=30,
        iterations=100,
        rho=0.1,            # evaporation rate in (0, 1)
        alpha=1.0,          # pheromone influence
        beta=2.0,           # heuristic influence
        tau0=1.0,           # initial pheromone level
        weights=None,       # per-objective weights; None -> uniform
        archive_size=100,
        tw_mode="soft",
    )

    EPS = 1e-9

    def _p(self, key: str) -> Any:
        v = self.params.get(key, self.DEFAULTS[key])
        return self.DEFAULTS[key] if v is None and key not in ("weights",) else v

    def _setup(self):
        """Allocate pheromone matrices and the heuristic desirability matrix."""
        n = self.instance.size
        m = len(self.objective_names)        # 2 in hard mode, 3 in soft mode
        tau0 = float(self._p("tau0"))

        self.tau = np.full((m, n, n), tau0, dtype=float) # pheromone matrix per objective

        # heuristic desirability eta[i, j] = 1 / d(i, j)
        with np.errstate(divide="ignore"):
            eta = 1.0 / np.where(self.dist > self.EPS, self.dist, np.inf)
        np.fill_diagonal(eta, 0.0)
        self.eta = eta

        w = self.params.get("weights")
        self.weights = (
            np.full(m, 1.0 / m) if w is None else np.asarray(w, dtype=float)[:m]
        )
        if len(self.weights) != m:
            raise ValueError(
                f"weights must have one entry per objective ({m}), got {len(self.weights)}."
            )

    def candidate_customers(self, route: List[int], unvisited: Sequence[int]) -> List[int]:
        """Checks which unvisted customer can be appended to the route right now, given the respective constraints"""
        ok: List[int] = []
        for c in unvisited:
            m = evaluate_route(list(route) + [c], self.instance, self.dist)
            if not m.capacity_ok:
                continue
            if self.respect_tw and not m.time_ok:
                continue
            ok.append(c)
        return ok


    def transition_probabilities(self, i: int, candidates: Sequence[int]) -> np.ndarray:
        """The ant's probability distribution over the next customer.
            p(i,j) = [ sum_m w_m * (tau^m_{i,j})^alpha ] * (eta_{i,j})^beta
        """
        p = np.zeros(len(candidates), dtype=float)

        if not candidates: # caution: put into place so that construct_ant_solution does not break if list is empty
            return np.array([], dtype=float)
        
        for idx, j in enumerate(candidates):
            pheromone_term = np.sum(self.weights * (self.tau[:, i, j] ** self.alpha))
            heuristic_term = self.eta[i, j] ** self.beta
            p[idx] = pheromone_term * heuristic_term
        # Guard against division by zero
        if np.sum(p) == 0:
            p = np.ones(len(candidates), dtype=float) / len(candidates)
        else:
            p /= np.sum(p)
        return p

    def construct_ant_solution(self) -> Solution:
        """One ant builds a complete routing, serving every customer once."""
        unvisited = set(range(1, self.instance.size))
        routes = []

        while unvisited:
            route = []
            i = 0  # start at depot
            while True:
                candidates = self.candidate_customers(route, list(unvisited))
                if not candidates:
                    break  # nothing else fits -> close route
                p = self.transition_probabilities(i, candidates)
                r = self.rng.random()
                cum = 0.0
                j = candidates[-1]  # default to last candidate if something goes wrong
                for idx, prob in enumerate(p):
                    cum += prob
                    if r <= cum:
                        j = candidates[idx]
                        break
                route.append(j)
                unvisited.remove(j)
                i = j

            if not route:
                j = unvisited.pop()
                route.append(j)

            routes.append(route)

        return Solution(routes)

    def deposit(self, archive: Sequence[Solution]) -> None:
        """Evaporate, then reinforce edges used by archive solutions:

            tau^m_{i,j} <- (1 - rho) * tau^m_{i,j}  +  sum over
                           { s in A : (i,j) in s } of  1 / f_m(s)

        """
        self.tau *= (1 - self.rho)  # evaporate
        for s in archive:
            eval_vector = self.evaluate(s).vector(self.objective_names)
            for m, f_m in enumerate(eval_vector):
                deposit_amount = 1.0 / (f_m + self.EPS)  # avoid division by zero
                for route in s.routes:
                    edges = [(0, route[0])] + [(route[i], route[i + 1]) for i in range(len(route) - 1)] + [(route[-1], 0)]
                    for i, j in edges:
                        self.tau[m, i, j] += deposit_amount

    def _solve(self) -> List[Solution]:
        self.alpha = float(self._p("alpha"))
        self.beta = float(self._p("beta"))
        self.rho = float(self._p("rho"))
        n_ants = int(self._p("n_ants"))
        iterations = int(self._p("iterations"))
        archive_size = int(self._p("archive_size"))

        self._setup()

        archive: List[Solution] = []
        ants: List[Solution] = [] 

        for _ in range(iterations):
            # each ant constructs a solution.
            ants = [self.construct_ant_solution() for _ in range(n_ants)]

            # evaluate and update the Pareto archive.
            archive = self.update_archive(archive, ants, max_size=archive_size)

            # evaporate + deposit from the archive.
            if archive:
                self.deposit(archive)

            self.log_generation(archive or ants)

        # return the Pareto archive.
        return archive if archive else ants
