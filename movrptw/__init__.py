"""MOVRPTW: a modular framework for benchmarking metaheuristics on the
Multi-Objective Vehicle Routing Problem with Time Windows."""

from __future__ import annotations

from .instance import Instance, Node, parse_solomon, parse_solomon_str
from .solution import (
    AVAILABLE_OBJECTIVES,
    Evaluation,
    Solution,
    evaluate,
    evaluate_route,
)
from .config import ExperimentConfig, load_config
from .metrics import hypervolume, run_metrics, score_fronts
from .solvers import (
    MetaheuristicSolver,
    SolverResult,
    available_solvers,
    get_solver,
    register_solver,
)
from .experiment import run_experiment, run_from_config_file

__version__ = "0.1.0"

__all__ = [
    "Instance",
    "Node",
    "parse_solomon",
    "parse_solomon_str",
    "Solution",
    "Evaluation",
    "evaluate",
    "evaluate_route",
    "AVAILABLE_OBJECTIVES",
    "ExperimentConfig",
    "load_config",
    "hypervolume",
    "score_fronts",
    "run_metrics",
    "MetaheuristicSolver",
    "SolverResult",
    "register_solver",
    "get_solver",
    "available_solvers",
    "run_experiment",
    "run_from_config_file",
    "__version__",
]
