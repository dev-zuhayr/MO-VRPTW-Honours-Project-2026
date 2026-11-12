#!/usr/bin/env python3
"""Expand a tuning spec into one experiment config per parameter combination.

The framework already expands an algorithm ``grid`` into parameter sets, but it
runs the whole grid in one process. On a cluster you want the grid spread over a
Slurm job array instead, so this script does the expansion up front and writes
one config file per combination:

    hpc/generated/<run-name>/
        shard_0000.yaml ... shard_00NN.yaml   one parameter combination each
        _all.yaml                             the whole grid, for the collect job

Array task ``k`` runs ``shard_%04d.yaml``. Every shard writes its results to its
own checkpoint directory, and the collect job hardlinks all of them into one
place and runs ``_all.yaml``, which finds every task already checkpointed and so
just writes the merged workbook.

That merge step is not optional bookkeeping. Hypervolume is normalised by the
pooled ideal/nadir of the runs written in one pass, so per-shard workbooks are
each on their own scale and cannot be compared. Only the collect job's workbook
ranks the parameter combinations against each other.

Usage
-----
    python hpc/make_configs.py hpc/specs/spea2.yaml --tw-mode hard --n-jobs 8
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Mirrors movrptw/tw.py. Duplicated deliberately so this script runs on a login
# node without the package installed; movrptw.config.validate() re-checks the
# pairing when the config is actually loaded.
OBJECTIVES = {
    "hard": ["total_distance", "num_vehicles"],
    "soft": ["total_distance", "num_vehicles", "total_tardiness"],
}


def load_spec(path: Path) -> Dict:
    spec = yaml.safe_load(path.read_text())
    for key in ("algorithm", "instances", "grid"):
        if key not in spec:
            raise SystemExit(f"{path}: spec is missing required key '{key}'.")
    return spec


def check_instances(instances: List[str]) -> None:
    """Fail on the login node rather than in 90 array tasks.

    Instance paths are relative to the repo root and are not opened until a
    solver actually runs, so a missing file surfaces as a wall of failed jobs
    hours later. Only six instances ship with the repo; the rest are downloaded.
    """
    missing = [p for p in instances if not (REPO_ROOT / p).is_file()]
    if not missing:
        return
    shown = "\n  ".join(missing[:10])
    more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
    raise SystemExit(
        f"{len(missing)} of {len(instances)} instance files are missing:\n  "
        f"{shown}{more}\n\n"
        "Only six instances ship with the repo. Fetch the rest first:\n"
        "    python scripts/download_instances.py --all"
    )


def combinations(grid: Dict[str, List]) -> List[Dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def base_config(spec: Dict, tw_mode: str, n_jobs: int, run_dir: Path) -> Dict:
    """The settings every shard shares. Paths are absolute so the sbatch scripts
    can be launched from anywhere."""
    return {
        "objectives": spec.get("objectives") or OBJECTIVES[tw_mode],
        "instances": list(spec["instances"]),
        "n_runs": int(spec.get("n_runs", 5)),
        "base_seed": int(spec.get("base_seed", 12345)),
        "n_jobs": int(n_jobs),
        "append": False,
        "distance_rounding": spec.get("distance_rounding"),
        "max_customers": spec.get("max_customers"),
    }


def write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="Tuning spec YAML (see hpc/specs/).")
    ap.add_argument("--tw-mode", choices=("hard", "soft"),
                    help="Override the spec's tw_mode. Objectives follow automatically.")
    ap.add_argument("--n-jobs", type=int, default=8,
                    help="Workers per shard. MUST match sbatch --cpus-per-task.")
    ap.add_argument("--run-name", help="Defaults to <algorithm>_<tw_mode>.")
    ap.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs",
                    help="Where run directories live. Put this on scratch on a cluster.")
    ap.add_argument("--force", action="store_true",
                    help="Delete an existing generated-config directory first. "
                         "Checkpoints and workbooks are left alone.")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    check_instances(list(spec["instances"]))
    algorithm = spec["algorithm"]
    tw_mode = args.tw_mode or spec.get("tw_mode", "hard")
    run_name = args.run_name or f"{algorithm}_{tw_mode}"

    run_dir = (args.runs_root / run_name).resolve()
    config_dir = run_dir / "configs"
    if config_dir.exists() and args.force:
        shutil.rmtree(config_dir)

    fixed = dict(spec.get("fixed", {}))
    fixed["tw_mode"] = tw_mode
    grid = dict(spec["grid"])
    combos = combinations(grid)

    base = base_config(spec, tw_mode, args.n_jobs, run_dir)

    for k, combo in enumerate(combos):
        cfg = dict(base)
        cfg["output"] = str(run_dir / "workbooks" / f"shard_{k:04d}.xlsx")
        cfg["checkpoint_dir"] = str(run_dir / "checkpoints" / f"shard_{k:04d}")
        # fixed + combo, never a grid: this shard is exactly one parameter set,
        # and the merged param dict matches the one _all.yaml produces, so both
        # generate identical checkpoint filenames.
        cfg["algorithms"] = [{"name": algorithm, "fixed": {**fixed, **combo}}]
        write_yaml(config_dir / f"shard_{k:04d}.yaml", cfg)

    all_cfg = dict(base)
    all_cfg["output"] = str(run_dir / f"tuning_{run_name}.xlsx")
    all_cfg["checkpoint_dir"] = str(run_dir / "checkpoints" / "merged")
    all_cfg["algorithms"] = [{"name": algorithm, "fixed": fixed, "grid": grid}]
    write_yaml(config_dir / "_all.yaml", all_cfg)

    # Parsed by submit.sh — keep the key=value shape.
    print(f"RUN_NAME={run_name}")
    print(f"RUN_DIR={run_dir}")
    print(f"CONFIG_DIR={config_dir}")
    print(f"NSHARDS={len(combos)}")
    print(f"RUNS_PER_SHARD={len(spec['instances']) * int(spec.get('n_runs', 5))}")


if __name__ == "__main__":
    main()
