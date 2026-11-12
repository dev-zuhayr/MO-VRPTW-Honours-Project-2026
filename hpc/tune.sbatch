#!/bin/bash
# ============================================================================
# UCT HPC (SLURM) — MOVRPTW parameter tuning.
# One array task = one parameter combination = one shard config.
#
# Submitted by hpc/submit.sh, which supplies --array, --cpus-per-task and an
# absolute --output path. To submit by hand:
#
#   sbatch --array=0-35%6 --cpus-per-task=20 hpc/tune.sbatch <RUN_DIR>
# ============================================================================
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1                 # joblib (loky) is node-local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20        # MUST equal n_jobs in the shard config
#SBATCH --time=48:00:00           # full 100-customer instances; trim after a
                                  # pilot. A timeout is recoverable: checkpoints
                                  # survive, so resubmitting resumes.
#SBATCH --job-name="movrptw-tune"
#SBATCH --mail-user=sprlaw001@myuct.ac.za
#SBATCH --mail-type=END,FAIL      # no ARRAY_TASKS -> one mail for the whole array
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

RUN_DIR="${1:?usage: sbatch tune.sbatch <RUN_DIR>}"
CONFIG="$RUN_DIR/configs/$(printf 'shard_%04d.yaml' "${SLURM_ARRAY_TASK_ID}")"

# SLURM_SUBMIT_DIR, not the script's own path: Slurm copies the batch script to
# a spool directory, so $BASH_SOURCE does not point into the repo.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

[[ -f "$CONFIG" ]] || { echo "No config for array index ${SLURM_ARRAY_TASK_ID}: $CONFIG" >&2; exit 1; }

module load python/miniconda3-py3.12
source "$HOME/movrptw-venv/bin/activate"

# joblib already forks one worker per CPU; letting BLAS spawn threads inside
# each worker oversubscribes the allocation and slows everything down.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg          # scripts/run_experiment.py imports matplotlib
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "alloc_cpus=${SLURM_CPUS_PER_TASK} config=$CONFIG"
echo "start=$(date --iso-8601=seconds)"

# No --no-resume and no --plots: checkpoints make a requeued or timed-out task
# cheap to restart, and per-shard plots are noise during tuning.
python scripts/run_experiment.py \
    --config "$CONFIG" \
    --logfile "$RUN_DIR/logs/shard_$(printf '%04d' "${SLURM_ARRAY_TASK_ID}").log"

echo "end=$(date --iso-8601=seconds)"
