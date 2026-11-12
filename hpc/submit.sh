#!/bin/bash
# Generate shard configs, submit the tuning array, chain the collect job.
#
#   ./hpc/submit.sh hpc/specs/spea2.yaml
#   ./hpc/submit.sh hpc/specs/moaco.yaml --tw-mode soft
#   ./hpc/submit.sh hpc/specs/moaco.yaml --pilot     # one shard, time it first
#
# --pilot submits array index 0 only and skips the collect job. Use it to
# measure how long one parameter combination takes, then set --time in
# hpc/tune.sbatch to roughly 2x that before launching the full sweep.

set -euo pipefail

# ---- SITE SETTINGS (UCT HPC / ada) ----------------------------------------
CPUS=20           # workers per shard. Passed to BOTH --cpus-per-task and the
                  # config's n_jobs, so the two cannot drift apart. There is no
                  # point exceeding (instances x n_runs) = the runs in a shard.
THROTTLE=6        # max array tasks running at once. Footprint is
                  # THROTTLE x CPUS cores = 120 here. Raise only if your QoS
                  # allows it: check `sacctmgr show qos` / ask the admins.
RUNS_ROOT="/scratch/$USER/movrptw/runs"   # never the home quota
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPEC="${1:?usage: ./hpc/submit.sh <spec.yaml> [--tw-mode hard|soft] [--pilot]}"
shift

PILOT=0
GEN_ARGS=()
while (( $# )); do
    case "$1" in
        --pilot) PILOT=1; shift ;;
        --tw-mode) GEN_ARGS+=(--tw-mode "$2"); shift 2 ;;
        --run-name) GEN_ARGS+=(--run-name "$2"); shift 2 ;;
        --force) GEN_ARGS+=(--force); shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$RUNS_ROOT" ]] && GEN_ARGS+=(--runs-root "$RUNS_ROOT")

# ---- generate ------------------------------------------------------------
GEN_OUT="$(python hpc/make_configs.py "$SPEC" --n-jobs "$CPUS" "${GEN_ARGS[@]}")"
echo "$GEN_OUT"

RUN_DIR="$(sed -n 's/^RUN_DIR=//p'  <<<"$GEN_OUT")"
NSHARDS="$(sed -n 's/^NSHARDS=//p'  <<<"$GEN_OUT")"
RUN_NAME="$(sed -n 's/^RUN_NAME=//p' <<<"$GEN_OUT")"

mkdir -p logs "$RUN_DIR/logs"

if (( PILOT )); then
    ARRAY="0-0"
else
    ARRAY="0-$((NSHARDS - 1))%${THROTTLE}"
fi

echo
echo "run      : $RUN_NAME"
echo "run dir  : $RUN_DIR"
echo "array    : $ARRAY   (${NSHARDS} parameter combinations)"
echo "cpus     : $CPUS per shard"
echo

# ---- submit --------------------------------------------------------------
TUNE_ID="$(sbatch --parsable \
    --array="$ARRAY" \
    --cpus-per-task="$CPUS" \
    --output="$RUN_DIR/logs/tune-%A_%a.out" \
    --error="$RUN_DIR/logs/tune-%A_%a.err" \
    hpc/tune.sbatch "$RUN_DIR")"
echo "submitted tuning array : $TUNE_ID"

if (( PILOT )); then
    echo
    echo "Pilot only — no collect job. Time it with:"
    echo "    sacct -j $TUNE_ID --format=JobID,Elapsed,MaxRSS,State"
    exit 0
fi

# afterany, not afterok: if a few shards time out, the remaining results are
# still worth aggregating, and the missing ones can be resubmitted into the same
# checkpoint directories and collected again.
COLLECT_ID="$(sbatch --parsable \
    --dependency=afterany:"$TUNE_ID" \
    --kill-on-invalid-dep=yes \
    --output="$RUN_DIR/logs/collect-%j.out" \
    --error="$RUN_DIR/logs/collect-%j.err" \
    hpc/collect.sbatch "$RUN_DIR")"
echo "submitted collect job  : $COLLECT_ID (after $TUNE_ID)"
echo
echo "watch:   squeue -u \$USER"
echo "result:  $RUN_DIR/tuning_${RUN_NAME}.xlsx"
