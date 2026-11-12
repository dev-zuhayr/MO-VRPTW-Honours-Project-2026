"""Rerun the Gurobi epsilon-constraint baseline at 25 or 50 customers and
merge the result back into that size's
``ResultsData/<N> customers final run/results_hard.xlsx``.

Why a rerun (rather than a post-hoc recompute like
``scripts/add_floor1_gap.py`` does for ALNS/MOPSO): the original Gurobi run's
routes were never written into ``results_hard.xlsx`` -- only its ``summary``
rows were merged in -- so there is nothing to recompute a floor1 distance
from. This run also solves on the floor1 distance matrix (``GurobiSolver``
now does that at 25/50 customers, see ``movrptw/solvers/gurobi.py``), so the
MILP optimum is comparable to the 25/50-customer BKS tables.

What it touches: only the ``gurobi`` rows, on the ``runs`` / ``pareto`` /
``solutions`` / ``summary`` sheets. ALNS/MOPSO rows everywhere -- and the
gurobi rows' existing ``hv_mean`` / ``hv_std`` -- are left exactly as they
are. Two distance-gap columns are written for every gurobi row, mirroring
the ALNS/MOPSO convention:

The reported point is the **min-distance feasible Pareto point** from
``res.pareto`` -- the same selection rule ``summary_frame`` uses for
ALNS/MOPSO (sort runs by ``total_distance``). It is *not* ``res.best`` (the
scalarised pick), which can favour a fewest-vehicles Pareto corner and bury
a proven BKS-matching optimum sitting at a higher vehicle count.
``res.best`` is left untouched on the ``SolverResult`` for other consumers.

* ``gap_distance_pct``        -- reported route re-scored on the full-precision
                                 distance matrix, gap vs the size's BKS table.
* ``gap_distance_pct_floor1`` -- same route scored with each edge floored to
                                 1 dp (the literature convention the solve now
                                 targets).
* ``solver_status`` / ``mip_gap`` -- whether the reported point's (vehicles,
                                 floor1-distance) was proven optimal at *any*
                                 epsilon level in the sweep, and the MIP gap
                                 there. "time_limit" with a non-zero
                                 ``mip_gap`` means no sweep level proved this
                                 objective value -- an unconverged incumbent.

The full per-epsilon-level trace (the veh_min step plus every sweep level
attempted, each with its status / MIP gap / distance / vehicle count) is
written to a dedicated ``gurobi_sweep`` sheet, so "did v=3 for RC201 actually
prove anything or just time out" is answerable without a rerun.

Usage:
    python scripts/rerun_gurobi_hard.py <size:25|50>
        [--time-limit 150] [--n-jobs 10] [--threads 2]
        [--instances C101,R201,RC201] [--dry-run]

``--instances`` restricts the rerun to a subset (for a quick check); the
merge step is skipped unless every gurobi instance was rerun, so a subset
run only prints a table and writes nothing.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movrptw import bks
from movrptw.experiment import _load_instance
from movrptw.results import _format_sheet, _routes_str
from movrptw.solution import Solution, evaluate
from movrptw.solvers.gurobi import GurobiSolver

OBJ_HARD = ["total_distance", "num_vehicles"]
SEED = 12345
SHEET_ORDER = ["summary", "runs", "pareto", "reference", "solutions",
               "instances", "config", "gurobi_sweep"]

#: Extra gurobi-only columns added to the runs / summary sheets.
_STATUS_COLS = ["solver_status", "mip_gap"]

_RESULTS_DIR = {
    25: Path("ResultsData/25 customers final run"),
    50: Path("ResultsData/50 customers final run"),
}

#: Per-instance solve results are pickled here so a mid-run failure (the WLS
#: Gurobi license needs periodic network access and a dropped connection
#: aborts the whole joblib batch) doesn't throw away completed solves. A
#: re-run picks up only the instances still missing. Delete this dir to
#: force a clean rerun.
_CKPT_ROOT = Path("outputs/checkpoints_gurobi_hard")


def _instance_file(name: str) -> str:
    # Windows FS is case-insensitive; data/instances/ mixes "C101.txt" and
    # "c102.txt". Match either without caring.
    p = Path("data/instances") / f"{name}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p)


def _ckpt_path(size: int, name: str) -> Path:
    return _CKPT_ROOT / str(size) / f"{name}.pkl"


def solve_one(name: str, size: int, time_limit: float, threads: int,
              _attempts: int = 4) -> dict:
    """Run one Gurobi solve and return everything the merge step needs,
    keyed by plain (whitespace-stripped) instance name. The result is also
    pickled to a per-instance checkpoint so a later re-run can skip it.

    Retries a few times on transient Gurobi errors -- the WLS licence fetches
    a token over the network and a brief connectivity blip otherwise aborts
    the whole joblib batch.
    """
    import gurobipy as gp
    for attempt in range(1, _attempts + 1):
        try:
            return _solve_one_inner(name, size, time_limit, threads)
        except gp.GurobiError as exc:
            if attempt == _attempts:
                raise
            print(f"  {name}: GurobiError ({exc}); retry {attempt}/{_attempts - 1} "
                  f"in {15 * attempt}s", flush=True)
            time.sleep(15 * attempt)


def _solve_one_inner(name: str, size: int, time_limit: float, threads: int) -> dict:
    instance = _load_instance(_instance_file(name), None, size)
    full = instance.distance_matrix()  # full-precision (already cached)

    solver = GurobiSolver(
        instance, OBJ_HARD, {"time_limit": time_limit, "threads": threads}, seed=SEED
    )
    floor1 = solver.dist  # floor1 matrix the solve actually used

    t0 = time.perf_counter()
    res = solver.run()
    elapsed = time.perf_counter() - t0

    sweep_trace = [dict(e) for e in getattr(solver, "sweep_trace", [])]
    for e in sweep_trace:
        e["is_winner"] = False
        e["reproduces_reported"] = False

    # The reported gap columns come from the MIN-DISTANCE feasible Pareto
    # point, matching the ALNS/MOPSO summary rule (summary_frame sorts runs
    # by total_distance). res.best is the *scalarised* pick, which can favour
    # a fewest-vehicles Pareto corner and bury a proven BKS-matching optimum
    # sitting at a higher vehicle count -- see the C207 / RC201 sweep traces.
    # res.best stays untouched on the SolverResult for any other consumer.
    cand = []
    for pt in res.pareto:
        e = evaluate(Solution(pt.solution.routes), instance, full)
        if e.feasible:
            cand.append((e.total_distance, pt, e))
    if cand:
        _, rep_pt, ev_full = min(cand, key=lambda c: c[0])
        rep_sol = rep_pt.solution
    else:
        rep_sol = res.best.solution
        ev_full = evaluate(Solution(rep_sol.routes), instance, full)

    best_routes = rep_sol.routes
    ev_fl = evaluate(Solution(best_routes), instance, floor1)

    gap_full = bks.gap(name, ev_full.num_vehicles, ev_full.total_distance, size=size)
    gap_fl = bks.gap(name, ev_fl.num_vehicles, ev_fl.total_distance, size=size)

    # Status of the reported point: was this (vehicles, floor1-distance)
    # combination proven optimal at *any* epsilon level in the sweep? A
    # solution found first as a timed-out incumbent at eps=k but then proven
    # optimal at eps=k+1 (same distance) is a proven optimum -- report it as
    # such rather than as the timed-out level that happened to surface it.
    rep_veh = int(ev_fl.num_vehicles)
    rep_dist = float(ev_fl.total_distance)
    matches = [e for e in sweep_trace
               if e["phase"] == "sweep" and e["feasible"]
               and e["num_vehicles"] == rep_veh
               and e["distance"] is not None
               and abs(e["distance"] - rep_dist) < 0.05]
    if matches:
        opt = [e for e in matches if e["status_name"] == "optimal"]
        pick = opt[0] if opt else min(
            matches, key=lambda e: (e["mip_gap"] if e["mip_gap"] is not None else 9e9))
        winner_status = pick["status_name"]
        winner_mip_gap = pick["mip_gap"]
        winner_epsilon = pick["epsilon"]
        for e in sweep_trace:
            e["is_winner"] = e is pick
        # every level that reproduced the reported objective (not just the
        # one credited above) is flagged too, so the sheet shows how many
        # epsilon levels independently confirmed it.
        _match_ids = {id(e) for e in matches}
        for e in sweep_trace:
            e["reproduces_reported"] = id(e) in _match_ids
    else:  # fall back to the tag carried on the Solution object
        winner_status = getattr(rep_sol, "gurobi_status_name", None)
        winner_mip_gap = getattr(rep_sol, "gurobi_mip_gap", None)
        winner_epsilon = getattr(rep_sol, "gurobi_epsilon", None)

    pareto_pts = []
    for k, pt in enumerate(res.pareto):
        pev = evaluate(Solution(pt.solution.routes), instance, full)
        pareto_pts.append({
            "point": k,
            "total_distance": pev.total_distance,
            "num_vehicles": pev.num_vehicles,
            "feasible": bool(pev.feasible),
            "is_reported": pt.solution is rep_sol,
            "routes": _routes_str(pt.solution.routes),
        })

    out = {
        "instance": name,
        "runtime_s": round(elapsed, 4),
        "iterations": res.iterations,
        "feasible": bool(ev_full.feasible),
        "n_pareto": len(res.pareto),
        "num_vehicles": int(ev_full.num_vehicles),
        "total_distance_full": float(ev_full.total_distance),
        "total_distance_floor1": float(ev_fl.total_distance),
        "gap_vehicles_pct": gap_full["gap_vehicles_pct"],
        "gap_distance_pct": gap_full["gap_distance_pct"],
        "gap_distance_pct_floor1": gap_fl["gap_distance_pct"],
        "bks_vehicles": gap_full["bks_vehicles"],
        "bks_distance": gap_full["bks_distance"],
        "solver_status": winner_status,
        "mip_gap": winner_mip_gap,
        "winner_epsilon": winner_epsilon,
        "routes_str": _routes_str(best_routes),
        "pareto_pts": pareto_pts,
        "sweep_trace": sweep_trace,
        # For contrast in the console table only: what res.best (the
        # scalarised pick) would have reported, had we used it.
        "scalarised_best_vehicles": int(res.best.objectives.get("num_vehicles", 0)),
        "scalarised_best_distance_fl1": float(res.best.objectives.get("total_distance", 0.0)),
    }
    cp = _ckpt_path(size, name)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump(out, fh)
    return out


def _load_checkpoints(size: int, names: list[str]) -> dict:
    done = {}
    for name in names:
        cp = _ckpt_path(size, name)
        if cp.exists():
            try:
                with open(cp, "rb") as fh:
                    done[name] = pickle.load(fh)
            except Exception:
                pass
    return done


def _stripped(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def merge_into_workbook(path: Path, solved: dict, size: int,
                        params_str: str) -> None:
    sheets = pd.read_excel(path, sheet_name=None)
    runs, pareto, solutions, summary = (
        sheets["runs"], sheets["pareto"], sheets["solutions"], sheets["summary"]
    )

    old_gur = summary[_stripped(summary["algorithm"]) == "gurobi"].copy()
    old_gur["_key"] = _stripped(old_gur["instance"])
    hv_by_inst = dict(zip(old_gur["_key"], old_gur.get("hv_mean")))
    hvstd_by_inst = dict(zip(old_gur["_key"], old_gur.get("hv_std")))

    # ---- runs sheet: append one gurobi row per instance ---------------- #
    runs_rows = []
    for name, s in solved.items():
        runs_rows.append({
            "instance": name, "algorithm": "gurobi", "seed": SEED,
            "runtime_s": s["runtime_s"], "iterations": s["iterations"],
            "feasible": s["feasible"], "n_pareto": s["n_pareto"],
            "total_distance": s["total_distance_full"],
            "num_vehicles": s["num_vehicles"], "params": params_str,
            "bks_vehicles": s["bks_vehicles"], "bks_distance": s["bks_distance"],
            "gap_vehicles_pct": s["gap_vehicles_pct"],
            "gap_distance_pct": s["gap_distance_pct"],
            "hv": hv_by_inst.get(name),
            "gap_distance_pct_floor1": s["gap_distance_pct_floor1"],
            "solver_status": s["solver_status"],
            "mip_gap": s["mip_gap"],
        })
    runs_new = pd.concat([runs, pd.DataFrame(runs_rows)], ignore_index=True)
    runs_new = runs_new[list(runs.columns)
                        + [c for c in _STATUS_COLS if c not in runs.columns]]

    # ---- solutions sheet --------------------------------------------- #
    sol_rows = [{
        "instance": name, "algorithm": "gurobi", "seed": SEED,
        "feasible": s["feasible"], "num_vehicles": s["num_vehicles"],
        "total_distance": round(s["total_distance_full"], 3),
        "routes": s["routes_str"],
    } for name, s in solved.items()]
    sol_new = pd.concat([solutions, pd.DataFrame(sol_rows)], ignore_index=True)
    sol_new = sol_new[solutions.columns]

    # ---- pareto sheet ---------------------------------------------- #
    par_rows = []
    for name, s in solved.items():
        for pt in s["pareto_pts"]:
            par_rows.append({
                "instance": name, "algorithm": "gurobi", "seed": SEED,
                "point": pt["point"], "total_distance": pt["total_distance"],
                "num_vehicles": pt["num_vehicles"], "routes": pt["routes"],
            })
    par_new = pd.concat([pareto, pd.DataFrame(par_rows)], ignore_index=True)
    par_new = par_new[pareto.columns]

    # ---- summary sheet: update gurobi rows in place ------------------- #
    summary_new = summary.copy()
    for c in _STATUS_COLS:
        if c not in summary_new.columns:
            summary_new[c] = pd.NA
    is_gur = _stripped(summary_new["algorithm"]) == "gurobi"
    key = _stripped(summary_new["instance"])
    for idx in summary_new.index[is_gur]:
        name = key[idx]
        s = solved.get(name)
        if s is None:
            raise RuntimeError(f"summary has gurobi/{name!r} but it was not rerun")
        d = s["total_distance_full"]
        summary_new.loc[idx, [
            "total_distance_mean", "total_distance_std",
            "total_distance_min", "total_distance_max",
            "num_vehicles_mean", "num_vehicles_std",
            "num_vehicles_min", "num_vehicles_max",
            "runtime_s_mean", "runtime_s_std", "success_rate",
            "gap_vehicles_pct", "gap_distance_pct", "gap_distance_pct_floor1",
            "solver_status", "mip_gap",
        ]] = [
            d, 0.0, d, d,
            s["num_vehicles"], 0.0, s["num_vehicles"], s["num_vehicles"],
            s["runtime_s"], 0.0, 1.0 if s["feasible"] else 0.0,
            s["gap_vehicles_pct"], s["gap_distance_pct"],
            s["gap_distance_pct_floor1"],
            s["solver_status"], s["mip_gap"],
        ]
        # hv_mean / hv_std deliberately left untouched.

    # ---- gurobi_sweep sheet: full per-epsilon-level trace ------------- #
    sweep_rows = []
    for name, s in solved.items():
        for e in s["sweep_trace"]:
            sweep_rows.append({"instance": name, **e})
    sweep_new = pd.DataFrame(sweep_rows, columns=[
        "instance", "phase", "epsilon", "status", "status_name", "mip_gap",
        "sol_count", "runtime_s", "distance", "num_vehicles", "feasible",
        "in_pool", "is_winner", "reproduces_reported",
    ])
    old_sweep = sheets.get("gurobi_sweep")
    if old_sweep is not None and not old_sweep.empty:
        keep = ~_stripped(old_sweep["instance"]).isin(set(solved))
        sweep_new = pd.concat([old_sweep[keep], sweep_new], ignore_index=True)

    sheets["runs"], sheets["pareto"] = runs_new, par_new
    sheets["solutions"], sheets["summary"] = sol_new, summary_new
    sheets["gurobi_sweep"] = sweep_new

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in SHEET_ORDER:
            df = sheets.get(sheet, pd.DataFrame())
            df.to_excel(writer, sheet_name=sheet, index=False)
            _format_sheet(writer.sheets[sheet], df)
    print(f"Merged {len(solved)} gurobi instances into {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", type=int, choices=(25, 50))
    ap.add_argument("--time-limit", type=float, default=150.0)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--instances", default=None,
                    help="comma-separated subset; merge is skipped unless all "
                         "gurobi instances are covered")
    ap.add_argument("--dry-run", action="store_true",
                    help="solve + print the table but never write the workbook")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing per-instance checkpoints and resolve all")
    args = ap.parse_args()

    path = _RESULTS_DIR[args.size] / "results_hard.xlsx"
    if not path.exists():
        sys.exit(f"not found: {path}")

    summary = pd.read_excel(path, sheet_name="summary")
    all_gur = sorted(_stripped(summary.loc[
        _stripped(summary["algorithm"]) == "gurobi", "instance"]).unique())
    if not all_gur:
        sys.exit(f"no gurobi rows in {path}'s summary sheet")

    targets = ([x.strip() for x in args.instances.split(",")]
               if args.instances else all_gur)
    params_str = f"threads={args.threads};time_limit={int(args.time_limit)}"

    done = {} if args.no_resume else _load_checkpoints(args.size, targets)
    pending = [n for n in targets if n not in done]
    print(f"size={args.size}  instances={len(targets)}  {params_str}  "
          f"n_jobs={args.n_jobs}  (resumed {len(done)}, pending {len(pending)})")

    t0 = time.perf_counter()
    fresh = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(solve_one)(name, args.size, args.time_limit, args.threads)
        for name in pending
    ) if pending else []
    solved = dict(done)
    solved.update({r["instance"]: r for r in fresh})
    print(f"solved {len(fresh)} fresh in {time.perf_counter() - t0:.0f}s "
          f"({len(solved)} total)\n")

    print("reported point = min-distance feasible Pareto point "
          "(scal = what res.best/scalariser would have picked)")
    hdr = f"{'instance':10s} {'veh':>4s} {'dist_full':>11s} {'dist_fl1':>10s} " \
          f"{'gap%':>8s} {'gap%_fl1':>9s} {'status':>11s} {'mipgap':>9s} " \
          f"{'scal(v@dfl1)':>14s} {'t(s)':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name in targets:
        s = solved[name]
        mg = "  n/a" if s["mip_gap"] is None else f"{s['mip_gap'] * 100:8.3f}%"
        scal = f"{s['scalarised_best_vehicles']}@{s['scalarised_best_distance_fl1']:.1f}"
        print(f"{name:10s} {s['num_vehicles']:4d} {s['total_distance_full']:11.3f} "
              f"{s['total_distance_floor1']:10.1f} {s['gap_distance_pct']:8.2f} "
              f"{s['gap_distance_pct_floor1']:9.2f} {str(s['solver_status']):>11s} "
              f"{mg:>9s} {scal:>14s} {s['runtime_s']:8.1f}")

    # Full per-epsilon-level sweep trace for every instance run.
    print("\n" + "=" * 78)
    print("per-level sweep trace  (phase/eps  status  mipgap  distance  veh  "
          "solcount  t)")
    print("=" * 78)
    for name in targets:
        print(f"\n{name}:")
        for e in solved[name]["sweep_trace"]:
            lvl = "veh_min" if e["phase"] == "veh_min" else f"eps={e['epsilon']}"
            mg = "     n/a" if e["mip_gap"] is None else f"{e['mip_gap'] * 100:7.3f}%"
            dist = "       -" if e["distance"] is None else f"{e['distance']:8.2f}"
            veh = "  -" if e["num_vehicles"] is None else f"{e['num_vehicles']:3d}"
            flags = []
            if e.get("is_winner"):
                flags.append("<-- REPORTED (status credited here)")
            elif e.get("reproduces_reported"):
                flags.append("(also reproduces reported point)")
            if e["phase"] == "sweep" and not e["in_pool"]:
                flags.append("not-in-pool")
            print(f"  {lvl:9s} {e['status_name']:11s} {mg:>8s}  dist={dist}  "
                  f"veh={veh}  sols={e['sol_count']:2d}  {e['runtime_s']:7.1f}s  "
                  f"{' '.join(flags)}")

    covered = set(solved) >= set(all_gur)
    if args.dry_run:
        print("\n--dry-run: workbook not written")
    elif not covered:
        missing = sorted(set(all_gur) - set(solved))
        print(f"\nsubset run ({len(missing)} gurobi instances not rerun); "
              f"workbook not written. Missing: {missing}")
    else:
        merge_into_workbook(path, solved, args.size, params_str)


if __name__ == "__main__":
    main()
