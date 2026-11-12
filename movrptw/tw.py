"""Time-window mode: the switch that lets every solver handle hard or soft
time windows.

The two modes are different problem shapes, not cosmetic variants:

— for hard, a time-window violation makes a solution *infeasible*, so every
valid solution has a total_tardiness = 0. Tardiness is then a dead
dimension and the problem has two objectives: total_distance and num_vehicles.

— for soft, a time-window violation is *allowed* and becomes objective f3,
giving three objectives: total_distance, num_vehicles and total_tardiness.

"""

from __future__ import annotations

from typing import Sequence

from .solution import Evaluation

HARD = "hard"
SOFT = "soft"
TW_MODES = (HARD, SOFT)

# Canonical objective lists for each mode.
OBJECTIVES_HARD = ["total_distance", "num_vehicles"]
OBJECTIVES_SOFT = ["total_distance", "num_vehicles", "total_tardiness"]

TOL = 1e-6


def admissible(ev: Evaluation, tw_mode: str) -> bool:
    """Return True if the solution may enter the population / archive.

    Capacity and coverage are always hard; time windows are hard only when
    ``tw_mode == "hard"``.

    ``Evaluation.feasible`` is deliberately not used: it always treats time
    windows as hard, which would discard every tardy solution in soft mode,
    the very solutions soft mode exists to explore.
    """
    _check_mode(tw_mode)
    ok = ev.capacity_violation <= TOL and ev.unserved == 0
    if tw_mode == HARD:
        ok = ok and ev.time_window_violation <= TOL
    return ok


def constraint_violation(ev: Evaluation, tw_mode: str) -> float:
    """Total hard-constraint violation; zero iff ``admissible``.

    Used to rank infeasible individuals so the search is pulled back toward
    the feasible region.
    """
    _check_mode(tw_mode)
    v = ev.capacity_violation + float(ev.unserved)
    if tw_mode == HARD:
        v += ev.time_window_violation
    return v


def objectives_for(tw_mode: str) -> list[str]:
    """Canonical objective list for a mode."""
    _check_mode(tw_mode)
    return list(OBJECTIVES_HARD if tw_mode == HARD else OBJECTIVES_SOFT)


def validate(tw_mode: str, objective_names: Sequence[str]) -> None:
    """Reject the two tw_mode / objective combinations that silently break a run."""
    _check_mode(tw_mode)
    has_tardiness = "total_tardiness" in objective_names
    if tw_mode == HARD and has_tardiness:
        raise ValueError(
            "tw_mode='hard' makes time-window violations infeasible, so "
            "'total_tardiness' is 0 for every admissible solution and is a "
            f"degenerate objective. Use objectives={OBJECTIVES_HARD}."
        )
    if tw_mode == SOFT and not has_tardiness:
        raise ValueError(
            "tw_mode='soft' permits time-window violations, so they must be "
            "measured. Add 'total_tardiness' to objectives, e.g. "
            f"objectives={OBJECTIVES_SOFT}."
        )


def _check_mode(tw_mode: str) -> None:
    if tw_mode not in TW_MODES:
        raise ValueError(f"tw_mode must be one of {TW_MODES}, got {tw_mode!r}.")
