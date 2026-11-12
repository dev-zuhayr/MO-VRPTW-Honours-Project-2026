"""SPEA2 — Strength Pareto Evolutionary Algorithm 2"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from ..solution import Solution
from .base import dominates, register_solver
from .construction import (
    bind_operators,
    nearest_neighbour_construct,
    random_neighbour,
    route_crossover,
)
from .mo_base import MOSolver, normalise, pairwise_distances


@register_solver("spea2")
class SPEA2(MOSolver):
    """Elitist MO evolutionary algorithm with strength-based fitness."""

    DEFAULTS = dict(
        pop_size=60,
        archive_size=40,
        generations=150,
        crossover_rate=0.9,
        #: Expected neighbourhood moves per child. <= 1 acts as a probability;
        #: above 1 it is local-search depth. See _mutation_moves.
        mutation_rate=0.3,
        k=None,          # None -> sqrt(pop_size + archive_size), the SPEA2 default
        tw_mode="soft",  
    )

    def _p(self, key: str) -> Any:
        v = self.params.get(key, self.DEFAULTS[key])
        return self.DEFAULTS[key] if v is None and key != "k" else v

    @staticmethod
    def strength(objs: np.ndarray) -> np.ndarray:
        """returns the strength: S(i) = number of solutions that individual i dominates."""
        n = objs.shape[0]
        s_i = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and dominates(objs[i], objs[j]):
                    s_i[i] += 1
        return s_i

    @staticmethod
    def raw_fitness(objs: np.ndarray, strength: np.ndarray) -> np.ndarray:
        """returns raw fitness: R(i) = sum of the strengths of every solution that dominates i.
        * non-dominated solutions will return 0, every dominated solution has R(i) >= 1.
        """
        n = objs.shape[0]
        r_i = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j and dominates(objs[j], objs[i]):
                    r_i[i] += strength[j]
        return r_i

    @staticmethod
    def density(objs: np.ndarray, k: int) -> np.ndarray:
        """D(i) = 1 / (sigma_i^k + 2), the density estimate.
        * density estimates help break ties between non-dominated solutions
        """
        n = objs.shape[0]
        if n == 0:
            return np.array([], dtype=float)
        k = min(k, n - 1)
        norm_objs = normalise(objs)
        dist_matrix = pairwise_distances(norm_objs)
        sigma_k = np.sort(dist_matrix, axis=1)[:, k]
        d_i = 1.0 / (sigma_k + 2.0)
        return d_i

    @staticmethod
    def truncate(objs: np.ndarray, selected: List[int], target: int) -> List[int]:
        """Truncation operator for environmental selection. Iteratively remove the individual with the smallest
        distance to its nearest neighbour.
        """
        if len(selected) <= target:
            return selected.copy()

        norm_objs = normalise(objs[selected])
        alive = selected.copy()

        while len(alive) > target:
            dist_matrix = pairwise_distances(norm_objs)
            np.fill_diagonal(dist_matrix, np.inf)
            sorted_d = np.sort(dist_matrix, axis=1)   
            to_remove_idx = min(range(len(alive)), key=lambda i: tuple(sorted_d[i]))
            alive.pop(to_remove_idx)
            norm_objs = np.delete(norm_objs, to_remove_idx, axis=0)

        return alive


    def fitness(self, objs: np.ndarray, k: int) -> np.ndarray:
        """F(i) = R(i) + D(i). Lower is better; F(i) < 1 iff non-dominated."""
        s = self.strength(objs)
        r = self.raw_fitness(objs, s)
        d = self.density(objs, k)
        return r + d

    def environmental_selection(
        self, objs: np.ndarray, fit: np.ndarray, archive_size: int
    ) -> List[int]:
        """Algorithm 4, step 5: build the next archive from the combined pool."""
        nondom = [i for i in range(len(objs)) if fit[i] < 1.0]

        if len(nondom) == archive_size:
            return nondom
        if len(nondom) < archive_size:
            # Fill the remaining slots with the best DOMINATED individuals by F.
            dominated = sorted(
                (i for i in range(len(objs)) if fit[i] >= 1.0),
                key=lambda i: fit[i],
            )
            return nondom + dominated[: archive_size - len(nondom)]
        # Too many non-dominated -> truncate the crowded ones away.
        return self.truncate(objs, nondom, archive_size)

    def _mutation_moves(self, mut_rate: float) -> int:
        """How many neighbourhood moves this child receives."""
        moves = int(mut_rate)
        if self.rng.random() < mut_rate - moves:
            moves += 1
        return moves

    def _binary_tournament(self, pool: List[Solution], fit_idx: dict) -> Solution:
        """binary tournament on F (lower wins)."""
        a, b = self.rng.randrange(len(pool)), self.rng.randrange(len(pool))
        return pool[a] if fit_idx[a] <= fit_idx[b] else pool[b]

    def _solve(self) -> List[Solution]:
        pop_size = int(self._p("pop_size"))
        archive_size = int(self._p("archive_size"))
        generations = int(self._p("generations"))
        cx_rate = float(self._p("crossover_rate"))
        mut_rate = float(self._p("mutation_rate"))
        k = self.params.get("k")
        if k is None:
            k = max(1, int(round(np.sqrt(pop_size + archive_size))))
        k = int(k)

        operators = bind_operators(self.instance, self.dist, respect_tw=self.respect_tw)

        pop: List[Solution] = [
            nearest_neighbour_construct(
                self.instance, self.dist, self.rng, respect_tw=self.respect_tw
            )
            for _ in range(pop_size)
        ]
        archive: List[Solution] = []

        for _ in range(generations):
            combined = pop + archive
            objs = self.objective_matrix(combined)
            fit = self.fitness(objs, k)

            keep = self.environmental_selection(objs, fit, archive_size)
            archive = [combined[i] for i in keep]
            archive_fit = {j: float(fit[i]) for j, i in enumerate(keep)}

            offspring: List[Solution] = []
            while len(offspring) < pop_size:
                p1 = self._binary_tournament(archive, archive_fit)
                p2 = self._binary_tournament(archive, archive_fit)
                child = (
                    route_crossover(
                        p1, p2, self.instance, self.dist, self.rng,
                        respect_tw=self.respect_tw,
                    )
                    if self.rng.random() < cx_rate
                    else p1.copy()
                )
                for _ in range(self._mutation_moves(mut_rate)):
                    child = random_neighbour(child, self.rng, operators)
                offspring.append(child)

            pop = offspring
            self.log_generation(archive)

        return archive if archive else pop
