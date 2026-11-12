"""Solver subpackage. Importing it registers every built-in solver."""

from .base import (  # noqa: F401
    MetaheuristicSolver,
    ScoredSolution,
    SolverResult,
    available_solvers,
    get_solver,
    register_solver,
)

# Importing each module triggers its @register_solver decorator.
from . import sa, nsga2, spea2, moaco, gurobi, mopso, alns, grasp  # noqa: F401,E402
from .mo_base import MOSolver  # noqa: F401,E402

__all__ = [
    "MetaheuristicSolver",
    "SolverResult",
    "ScoredSolution",
    "register_solver",
    "get_solver",
    "available_solvers",
]
