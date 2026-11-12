"""Matplotlib visualisation helpers for solutions and experiment output."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from .instance import Instance
from .solvers.base import ScoredSolution, SolverResult


def plot_routes(
    instance: Instance,
    routes: List[List[int]],
    *,
    title: Optional[str] = None,
    path: Optional[str | Path] = None,
):
    """Scatter customers and draw each vehicle route from/to the depot."""
    coords = instance.coords
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab20")

    for ri, route in enumerate(routes):
        xs = [coords[0, 0]] + [coords[c, 0] for c in route] + [coords[0, 0]]
        ys = [coords[0, 1]] + [coords[c, 1] for c in route] + [coords[0, 1]]
        ax.plot(xs, ys, "-o", color=cmap(ri % 20), markersize=4, linewidth=1.2,
                label=f"veh {ri + 1}")

    ax.scatter(coords[1:, 0], coords[1:, 1], c="#444", s=10, zorder=3)
    ax.scatter([coords[0, 0]], [coords[0, 1]], marker="s", s=120,
               c="#5E17EB", edgecolors="black", zorder=4, label="depot")
    ax.set_title(title or f"{instance.name}: {len(routes)} routes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if len(routes) <= 12:
        ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    return _finish(fig, path)


def plot_convergence(
    results: Sequence[SolverResult],
    *,
    title: str = "Convergence",
    path: Optional[str | Path] = None,
):
    """Overlay best-scalar-vs-iteration curves for several runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        if r.convergence:
            ax.plot(r.convergence, linewidth=1,
                    label=f"{r.algorithm} (seed {r.seed})")
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel("best scalarised objective")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, path)


def plot_pareto(
    pareto: Sequence[ScoredSolution],
    objective_names: Sequence[str],
    *,
    title: str = "Pareto front",
    path: Optional[str | Path] = None,
):
    """Scatter a 2-D Pareto front (first two objectives)."""
    if len(objective_names) < 2:
        raise ValueError("Need at least two objectives to plot a Pareto front.")
    ox, oy = objective_names[0], objective_names[1]
    pts = sorted(((p.objectives[ox], p.objectives[oy]) for p in pareto))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "-o", color="#5E17EB", markersize=6)
    ax.set_xlabel(ox)
    ax.set_ylabel(oy)
    ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, path)


def _finish(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig
