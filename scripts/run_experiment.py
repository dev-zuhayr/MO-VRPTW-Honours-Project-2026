#!/usr/bin/env python3
"""Run a MOVRPTW experiment from a configuration file.

Examples
--------
    python scripts/run_experiment.py --config configs/experiment.yaml
    python scripts/run_experiment.py --config configs/smoke.yaml --no-resume
    python scripts/run_experiment.py --config configs/experiment.yaml --plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movrptw import load_config  # noqa: E402
from movrptw.experiment import run_experiment  # noqa: E402
from movrptw.instance import parse_solomon  # noqa: E402
from movrptw import viz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a MOVRPTW benchmark experiment.")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing checkpoints and rerun everything.")
    ap.add_argument("--logfile", default="outputs/experiment.log")
    ap.add_argument("--plots", action="store_true",
                    help="Also write route/convergence/Pareto PNGs to outputs/plots/.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = run_experiment(cfg, resume=not args.no_resume, logfile=args.logfile)

    if args.plots and results:
        _write_plots(cfg, results)

    print(f"\nDone. {len(results)} runs written to {cfg.output}")


def _write_plots(cfg, results) -> None:
    out = Path("outputs/plots")
    out.mkdir(parents=True, exist_ok=True)
    instances = {p: parse_solomon(p) for p in cfg.instances}
    by_name = {parse_solomon(p).name: parse_solomon(p) for p in cfg.instances}

    # Best route map per instance (across all runs).
    best_per_instance: dict = {}
    for r in results:
        cur = best_per_instance.get(r.instance)
        if cur is None or r.best.objectives["total_distance"] < cur.best.objectives["total_distance"]:
            best_per_instance[r.instance] = r
    for name, r in best_per_instance.items():
        viz.plot_routes(by_name[name], r.best.routes,
                        title=f"{name} best ({r.algorithm}) "
                              f"v={int(r.best.objectives['num_vehicles'])} "
                              f"d={r.best.objectives['total_distance']:.0f}",
                        path=out / f"routes_{name}.png")

    # Convergence overlay (one representative run per algorithm on first instance).
    first = cfg.instances[0]
    first_name = parse_solomon(first).name
    subset = [r for r in results if r.instance == first_name]
    if subset:
        viz.plot_convergence(subset, title=f"Convergence on {first_name}",
                             path=out / f"convergence_{first_name}.png")

    # Pareto fronts for multi-objective runs.
    for r in results:
        if r.pareto and len(cfg.objectives) >= 2:
            viz.plot_pareto(r.pareto, cfg.objectives,
                            title=f"{r.algorithm} Pareto front: {r.instance}",
                            path=out / f"pareto_{r.instance}_{r.algorithm}_{r.seed}.png")
            break  # one example is enough for a script run

    print(f"Plots written to {out}")


if __name__ == "__main__":
    main()
