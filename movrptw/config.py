"""Experiment configuration: dataclasses + YAML/JSON loader.

A config defines *what* to run. The experiment runner expands every algorithm's
parameter grid into concrete parameter sets and runs each combination across
the requested instances and seeds.

Example YAML::

    output: outputs/results.xlsx
    append: false
    n_runs: 5
    base_seed: 12345
    n_jobs: 4
    timeout_s: 120
    objectives: [num_vehicles, total_distance]
    instances:
      - data/instances/C101.txt
      - data/instances/R101.txt
    algorithms:
      - name: nsga2
        grid:
          pop_size: [60]
          generations: [150]
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from .solution import AVAILABLE_OBJECTIVES


@dataclass
class AlgorithmSpec:
    name: str
    grid: Dict[str, List] = field(default_factory=dict)
    fixed: Dict = field(default_factory=dict)

    def param_sets(self) -> List[Dict]:
        """Cartesian product of the grid, merged with fixed params."""
        if not self.grid:
            return [dict(self.fixed)]
        keys = list(self.grid)
        combos = itertools.product(*(self.grid[k] for k in keys))
        return [{**self.fixed, **dict(zip(keys, combo))} for combo in combos]


@dataclass
class ExperimentConfig:
    instances: List[str]
    algorithms: List[AlgorithmSpec]
    objectives: List[str] = field(default_factory=lambda: ["num_vehicles", "total_distance"])
    n_runs: int = 5
    base_seed: int = 12345
    n_jobs: int = 1
    timeout_s: float | None = None
    output: str = "outputs/results.xlsx"
    append: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    distance_rounding: str | None = None
    max_customers: int | None = None

    def validate(self) -> None:
        bad = [o for o in self.objectives if o not in AVAILABLE_OBJECTIVES]
        if bad:
            raise ValueError(
                f"Unknown objective(s) {bad}. Available: {sorted(AVAILABLE_OBJECTIVES)}"
            )
        if not self.instances:
            raise ValueError("No instances specified.")
        if not self.algorithms:
            raise ValueError("No algorithms specified.")

        # Each tw_mode must be consistent with the objective list; otherwise
        # the run silently degenerates. See movrptw/tw.py for details.
        from .tw import validate as validate_tw

        for algo in self.algorithms:
            modes = algo.grid.get("tw_mode", [algo.fixed.get("tw_mode")])
            for mode in modes:
                if mode is not None:
                    validate_tw(str(mode), self.objectives)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    text = path.read_text()
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return from_dict(raw)


def from_dict(raw: Dict) -> ExperimentConfig:
    algos = [
        AlgorithmSpec(
            name=a["name"],
            grid=a.get("grid", {}) or {},
            fixed=a.get("fixed", {}) or {},
        )
        for a in raw["algorithms"]
    ]
    cfg = ExperimentConfig(
        instances=list(raw["instances"]),
        algorithms=algos,
        objectives=list(raw.get("objectives", ["num_vehicles", "total_distance"])),
        n_runs=int(raw.get("n_runs", 5)),
        base_seed=int(raw.get("base_seed", 12345)),
        n_jobs=int(raw.get("n_jobs", 1)),
        timeout_s=raw.get("timeout_s"),
        output=raw.get("output", "outputs/results.xlsx"),
        append=bool(raw.get("append", False)),
        checkpoint_dir=raw.get("checkpoint_dir", "outputs/checkpoints"),
        distance_rounding=raw.get("distance_rounding"),
        max_customers=raw.get("max_customers"),
    )
    cfg.validate()
    return cfg
