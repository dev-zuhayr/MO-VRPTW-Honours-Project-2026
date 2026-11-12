"""Pareto-approximation quality: the hypervolume indicator (HV).

HV is a *set* metric: it scores a whole approximation front rather than one
solution. That is what a front-quality research question needs — gap-to-BKS
scores a single point and says nothing about the convergence or the spread of
the front around it.

Every objective in this framework is minimised, so a front ``F`` is an
``(n, m)`` matrix of points to be pushed toward the origin. ``HV(F, r)`` is the
m-dimensional volume of the region dominated by ``F`` and bounded above by a
reference point ``r``::

    HV(F, r) = Lebesgue( union over f in F of the box [f, r] )

Bigger is better. It measures convergence *and* spread at once, and it is
Pareto-compliant: if ``F`` dominates ``F'`` then ``HV(F) > HV(F')``. Crucially
it needs no knowledge of the true Pareto front — only a reference point — which
is why it is the indicator of choice when no proven-optimal front exists to
compare against.

**Normalisation.** ``total_distance`` runs to the hundreds while
``num_vehicles`` is single digits, and HV multiplies edge lengths, so a raw HV
would be a distance-only volume. Each objective column is therefore min-max
scaled by the pooled ideal/nadir of the group being compared
(:class:`Bounds`) before the volume is taken, and the reference point sits just
past the nadir at ``ref_factor`` (1.1 by default) in every coordinate, so the
worst point in the group still encloses volume. HV consequently lands in
``[0, ref_factor ** m]``.

**Comparison groups.** Runs are only scored against each other within one
instance and one objective list (:func:`group_key`); the objective list also
separates hard from soft time-window runs. Since the scaling is pooled from the
runs in a group, HV values are comparable within a group and not across
instances, and a group must be rescored as a whole when runs are added — which
is what :func:`run_metrics` does.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from . import tw

if TYPE_CHECKING: 
    from .solvers.base import SolverResult

__all__ = [
    "DEFAULT_REF_FACTOR",
    "Bounds",
    "FrontQuality",
    "FrontComparison",
    "as_matrix",
    "pareto_filter",
    "bounds_of",
    "hypervolume",
    "score_fronts",
    "front_of",
    "group_key",
    "run_metrics",
]

#: Where the HV reference point sits relative to the normalised nadir (1.0).
DEFAULT_REF_FACTOR = 1.1

_TINY = 1e-12

Front = Sequence[Sequence[float]]

def as_matrix(front: Front | np.ndarray) -> np.ndarray:
    """Coerce a front to an ``(n, m)`` float matrix; an empty front is ``(0, 0)``."""
    objs = np.asarray(front, dtype=float)
    if objs.ndim == 2:
        return objs
    if objs.ndim == 1:
        return objs.reshape(0, 0) if objs.size == 0 else objs.reshape(1, -1)
    raise ValueError(f"Expected an (n, m) objective matrix, got shape {objs.shape}.")


def pareto_filter(front: Front | np.ndarray) -> np.ndarray:
    """Return the deduplicated non-dominated rows of ``front`` (minimisation)."""
    objs = as_matrix(front)
    if len(objs) == 0:
        return objs
    uniq = np.unique(objs, axis=0)
    keep = np.ones(len(uniq), dtype=bool)
    for i in range(len(uniq)):
        # Anything i dominates is also dominated by whatever dominates i, so a
        # point already discarded can be skipped without missing removals.
        if not keep[i]:
            continue
        dominated = np.all(uniq >= uniq[i], axis=1) & np.any(uniq > uniq[i], axis=1)
        keep &= ~dominated
    return uniq[keep]


@dataclass(frozen=True)
class Bounds:
    """Per-objective ideal (component-wise min) and nadir (component-wise max)."""

    ideal: np.ndarray
    nadir: np.ndarray

    @property
    def m(self) -> int:
        return int(np.asarray(self.ideal).size)

    def normalise(self, front: Front | np.ndarray) -> np.ndarray:
        """Min-max scale a front into ``[0, 1]^m``.

        A degenerate objective (every point identical, e.g. ``num_vehicles``
        when nothing varies) is mapped to 0 rather than dividing by zero.
        """
        objs = as_matrix(front)
        if len(objs) == 0:
            return objs
        span = self.nadir - self.ideal
        span = np.where(span < _TINY, 1.0, span)
        return (objs - self.ideal) / span

    def ref_point(self, factor: float = DEFAULT_REF_FACTOR) -> np.ndarray:
        """HV reference point in *normalised* space, just past the nadir."""
        return np.full(self.m, float(factor))


def bounds_of(fronts: Iterable[Front | np.ndarray]) -> Bounds:
    """Pooled ideal/nadir over several fronts."""
    stack = [f for f in (as_matrix(x) for x in fronts) if len(f)]
    if not stack:
        raise ValueError("Cannot compute bounds: every front is empty.")
    pooled = np.vstack(stack)
    return Bounds(ideal=pooled.min(axis=0), nadir=pooled.max(axis=0))

def hypervolume(front: Front | np.ndarray, ref_point: Sequence[float]) -> float:
    """Exact hypervolume of a minimisation front bounded by ``ref_point``.

    Points that do not strictly dominate the reference point in every
    coordinate enclose zero volume and are dropped; dominated and duplicate
    points cannot change the union's volume and are filtered out too, which is
    what lets the slicing recursion stay simple.
    """
    objs = as_matrix(front)
    ref = np.asarray(ref_point, dtype=float).ravel()
    if len(objs) == 0:
        return 0.0
    if objs.shape[1] != ref.size:
        raise ValueError(
            f"Front has {objs.shape[1]} objectives but the reference point has {ref.size}."
        )
    objs = objs[np.all(objs < ref, axis=1)]
    if len(objs) == 0:
        return 0.0
    return float(_hv(pareto_filter(objs), ref))


def _hv(objs: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume by slicing (HSO): recurse on m, sweep at m = 2."""
    m = ref.size
    if m == 1:
        return float(ref[0] - objs[:, 0].min())
    if m == 2:
        return _hv2(objs, ref)

    pts = objs[np.argsort(objs[:, -1])]
    upper = np.append(pts[1:, -1], ref[-1])
    vol = 0.0
    for k in range(len(pts)):
        depth = upper[k] - pts[k, -1]
        if depth <= 0.0:  # ties contribute no slab
            continue
        # Cross-section of the slab: everything already "open" in the last axis.
        vol += _hv(pts[: k + 1, :-1], ref[:-1]) * depth
    return vol


def _hv2(objs: np.ndarray, ref: np.ndarray) -> float:
    """2-D sweep: walk left to right, accumulating the staircase under ``ref``."""
    pts = objs[np.lexsort((objs[:, 1], objs[:, 0]))]
    vol = 0.0
    ceiling = ref[1]
    for x, y in pts:
        if y < ceiling:
            vol += (ref[0] - x) * (ceiling - y)
            ceiling = y
    return vol

@dataclass(frozen=True)
class FrontQuality:
    """HV of one front, measured in the normalised space of its group."""

    hv: float
    n_points: int


@dataclass(frozen=True)
class FrontComparison:
    """Result of scoring a group of fronts on a shared normalisation."""

    quality: Dict[Hashable, FrontQuality]
    bounds: Bounds
    ref_factor: float

    @property
    def ref_point(self) -> np.ndarray:
        return self.bounds.ref_point(self.ref_factor)


def score_fronts(
    fronts: Mapping[Hashable, Front | np.ndarray],
    *,
    ref_factor: float = DEFAULT_REF_FACTOR,
) -> FrontComparison:
    """Score every front in ``fronts`` on a common normalisation.

    The ideal/nadir are pooled from ``fronts`` themselves, so no point in the
    comparison ever falls outside ``[0, 1]``.

    Empty fronts get NaN and stay in the mapping, so callers can align the
    output with their input.
    """
    matrices = {k: as_matrix(v) for k, v in fronts.items()}
    pool: List[np.ndarray] = [f for f in matrices.values() if len(f)]
    bounds = bounds_of(pool)  # raises if the whole group is empty
    ref_point = bounds.ref_point(ref_factor)

    quality: Dict[Hashable, FrontQuality] = {}
    for key, front in matrices.items():
        if len(front) == 0:
            quality[key] = FrontQuality(hv=float("nan"), n_points=0)
            continue
        quality[key] = FrontQuality(
            hv=hypervolume(bounds.normalise(front), ref_point),
            n_points=len(front),
        )
    return FrontComparison(quality=quality, bounds=bounds, ref_factor=float(ref_factor))

def group_key(result: "SolverResult") -> Tuple[str, Tuple[str, ...]]:
    """Runs are only comparable within one instance and one objective set.

    The objective list also separates hard from soft time-window runs: soft mode
    carries ``total_tardiness`` and hard mode cannot (see :mod:`movrptw.tw`), so
    the two never share a normalisation.
    """
    return (result.instance, tuple(result.objective_names))


def front_of(result: "SolverResult") -> np.ndarray:
    """Objective matrix of a run's front, admissible points only.

    Two deliberate choices:

    * Inadmissible points are dropped. Multi-objective archives already hold
      only admissible solutions, but a scalarised solver can return a
      capacity-violating best, and one such point would sit below the whole
      front and wreck the normalisation.
    * A single-objective run has no archive, so its best solution is used as a
      one-point front. HV is still well defined on a singleton, which keeps
      GA/SA in the comparison instead of silently dropping them.
    """
    names = list(result.objective_names)
    mode = tw.SOFT if "total_tardiness" in names else tw.HARD
    points = result.pareto or [result.best]
    rows = [
        [p.objectives[n] for n in names]
        for p in points
        if tw.admissible(p.evaluation, mode)
    ]
    return np.asarray(rows, dtype=float).reshape(len(rows), len(names))


def run_metrics(
    results: Sequence["SolverResult"],
    *,
    ref_factor: float = DEFAULT_REF_FACTOR,
) -> Tuple[List[FrontQuality], Dict[Tuple[str, Tuple[str, ...]], FrontComparison]]:
    """Score every run's front, normalised per :func:`group_key`.

    Returns ``(per_run, per_group)``: a list of :class:`FrontQuality` aligned
    positionally with ``results``, and the :class:`FrontComparison` used for
    each group (its bounds and reference point are what make the numbers
    reproducible, so they are worth reporting alongside).

    A group in which no run produced a single admissible point is skipped, and
    its runs keep NaN metrics.
    """
    results = list(results)
    empty = FrontQuality(hv=float("nan"), n_points=0)
    per_run: List[FrontQuality] = [empty] * len(results)
    per_group: Dict[Tuple[str, Tuple[str, ...]], FrontComparison] = {}

    groups: Dict[Tuple[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    for i, r in enumerate(results):
        groups[group_key(r)].append(i)

    for key, idx in groups.items():
        fronts = {i: front_of(results[i]) for i in idx}
        try:
            comparison = score_fronts(fronts, ref_factor=ref_factor)
        except ValueError:  # nothing admissible anywhere in this group
            continue
        per_group[key] = comparison
        for i in idx:
            per_run[i] = comparison.quality[i]
    return per_run, per_group
