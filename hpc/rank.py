#!/usr/bin/env python3
"""Rank a tuning sweep's parameter sets and name the winner per algorithm.

    python hpc/rank.py /scratch/$USER/movrptw/runs/spea2_hard/tuning_spea2_hard.xlsx

Reads the merged workbook's ``runs`` sheet, aggregates hypervolume over seeds
and then over instances, and writes a ``ranking`` sheet back into the same file.

Why not just sort the ``summary`` sheet by ``hv_mean``: that sheet has one row
per instance x algorithm x params, so with 56 instances a 36-point sweep is
2,016 rows and no single row answers "which settings should I use". Ranking
across instances is the missing step.

Two aggregations, reported side by side:

* ``mean_hv``   — average HV across instances. HV is normalised per instance to
  [0, 1.1^m], so averaging is defensible, but an instance where every setting
  scores poorly still drags the mean around.
* ``mean_rank`` — rank the settings within each instance (1 = best), then
  average the ranks. Insensitive to instance difficulty, and the basis for a
  Friedman test if you want a significance claim in the write-up.

Sorted by ``mean_rank``. When the two disagree, say so in the write-up and
prefer ``mean_rank`` — the disagreement means one setting wins big on a few
instances and loses narrowly on many.

Runs that produced no admissible front have NaN hypervolume. They are excluded
from ``mean_hv`` and ranked last within their instance, and the ``coverage``
column reports the fraction of runs that produced a usable front — a setting
with a great mean_hv and coverage 0.4 is not a good setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rank_workbook(path: Path) -> pd.DataFrame:
    runs = pd.read_excel(path, sheet_name="runs")
    for col in ("algorithm", "params", "instance", "hv"):
        if col not in runs.columns:
            raise SystemExit(f"{path}: 'runs' sheet has no '{col}' column.")

    # Seeds -> one value per (algorithm, params, instance).
    per_instance = (
        runs.groupby(["algorithm", "params", "instance"], dropna=False)
        .agg(hv=("hv", "mean"), coverage=("hv", lambda s: s.notna().mean()))
        .reset_index()
    )

    # Rank settings within each (algorithm, instance); NaN sorts last.
    per_instance["rank"] = (
        per_instance.groupby(["algorithm", "instance"], dropna=False)["hv"]
        .rank(ascending=False, method="average", na_option="bottom")
    )

    ranking = (
        per_instance.groupby(["algorithm", "params"], dropna=False)
        .agg(
            mean_rank=("rank", "mean"),
            mean_hv=("hv", "mean"),
            std_hv=("hv", "std"),
            wins=("rank", lambda r: int((r == 1).sum())),
            coverage=("coverage", "mean"),
            n_instances=("instance", "nunique"),
        )
        .reset_index()
        .sort_values(["algorithm", "mean_rank"])
    )
    for col in ("mean_rank", "mean_hv", "std_hv", "coverage"):
        ranking[col] = ranking[col].round(4)
    return ranking


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workbook", type=Path, help="tuning_<run>.xlsx from the collect job")
    ap.add_argument("--top", type=int, default=10, help="rows to print per algorithm")
    ap.add_argument("--no-write", action="store_true",
                    help="print only; do not add a 'ranking' sheet to the workbook")
    args = ap.parse_args()

    if not args.workbook.is_file():
        raise SystemExit(f"No such workbook: {args.workbook}")

    ranking = rank_workbook(args.workbook)

    for algo, block in ranking.groupby("algorithm"):
        # ASCII only: Slurm log files and Windows consoles both mangle non-ASCII.
        print(f"\n=== {algo}: {len(block)} parameter sets, "
              f"{int(block['n_instances'].max())} instances ===")
        print(block.drop(columns=["algorithm"]).head(args.top).to_string(index=False))
        best = block.iloc[0]
        print(f"\nbest by mean_rank: {best['params']}")
        by_hv = block.sort_values("mean_hv", ascending=False).iloc[0]
        if by_hv["params"] != best["params"]:
            print(f"NOTE: mean_hv prefers a different setting: {by_hv['params']}")
            print("      Prefer mean_rank, and report the disagreement.")
        if best["coverage"] < 0.999:
            print(f"WARNING: coverage {best['coverage']:.2f} — this setting failed to "
                  "produce an admissible front on some runs.")

    if not args.no_write:
        with pd.ExcelWriter(args.workbook, engine="openpyxl",
                            mode="a", if_sheet_exists="replace") as writer:
            ranking.to_excel(writer, sheet_name="ranking", index=False)
        print(f"\nwrote 'ranking' sheet to {args.workbook}")


if __name__ == "__main__":
    main()
