#!/bin/bash
# ============================================================================
# UCT HPC (SLURM) — merge every shard's checkpoints into ONE workbook.
#
#   sbatch hpc/collect.sbatch <RUN_DIR>
#
# This is the only workbook whose hypervolume column is meaningful across
# parameter combinations: HV is normalised by the pooled ideal/nadir of the runs
# written in a single pass, so per-shard workbooks each sit on their own scale.
# Comparing shard_0003.xlsx against shard_0017.xlsx is meaningless; comparing
# rows inside tuning_<run>.xlsx is the point of the sweep.
#
# Runs no solvers of its own: every task is already checkpointed, so the runner
# finds 0 pending and goes straight to aggregation. Safe to rerun at any time.
# ============================================================================
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name="movrptw-collect"
#SBATCH --mail-user=sprlaw001@myuct.ac.za
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

RUN_DIR="${1:?usage: sbatch collect.sbatch <RUN_DIR>}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load python/miniconda3-py3.12
source "$HOME/movrptw-venv/bin/activate"

export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

MERGED="$RUN_DIR/checkpoints/merged"
mkdir -p "$MERGED"

# Checkpoint filenames encode instance|algorithm|params|seed, so shards can
# never collide. Hardlink to avoid duplicating the pickles; fall back to a copy
# on filesystems that refuse cross-directory links.
shopt -s nullglob
shards=("$RUN_DIR"/checkpoints/shard_*/*.pkl)
if (( ${#shards[@]} == 0 )); then
    echo "No shard checkpoints under $RUN_DIR/checkpoints — did the array run?" >&2
    exit 1
fi
cp -l "${shards[@]}" "$MERGED"/ 2>/dev/null || cp -f "${shards[@]}" "$MERGED"/
echo "merged ${#shards[@]} checkpoints into $MERGED"

# The log reports how many of the expected tasks actually landed. A short array
# (some tasks timed out or were cancelled) still produces a valid workbook of
# whatever finished — which is why submit.sh chains this with afterany.
python scripts/run_experiment.py \
    --config "$RUN_DIR/configs/_all.yaml" \
    --logfile "$RUN_DIR/logs/collect.log"

echo "workbook written under $RUN_DIR"
