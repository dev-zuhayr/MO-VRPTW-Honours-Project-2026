#!/bin/bash
#SBATCH --job-name=movrptw_gurobi_hard_25
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/lvrmig001/movrptw/logs/gurobi-hard-25-%j.out
#SBATCH --error=/scratch/lvrmig001/movrptw/logs/gurobi-hard-25-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lvrmig001@myuct.ac.za

# Re-run the Gurobi epsilon-constraint baseline at 25 customers on the
# floor1-consistent distance matrix (movrptw/solvers/gurobi.py) and merge
# the result -- runs / pareto / solutions / summary rows + routes +
# gap_distance_pct_floor1 + solver_status / mip_gap + the gurobi_sweep
# trace sheet -- back into
#   ResultsData/25 customers final run/results_hard.xlsx
#
# time_limit is 1000s per epsilon level (NOT the 150s used for local
# smoke-testing): the local run showed instances like C207 do not converge
# even at 1000s, so anything less has no chance on the hard wide-window
# C2/R2/RC2 cases.
#
# Prerequisites on the cluster (all relative to the workdir below):
#   * data/instances/*.txt                         (56 Solomon files)
#   * "ResultsData/25 customers final run/results_hard.xlsx"
#         -- the ALNS/MOPSO results this merges the gurobi rows into;
#            rsync it up before submitting, rsync it back down after.
#   * outputs/checkpoints_gurobi_hard/25/  is created automatically; each
#         instance is checkpointed as it finishes, so if the job hits the
#         wall-time limit before all 56 are done, just resubmit -- it
#         resumes and only the merge (which needs all 56) is deferred.
#
# --n-jobs 10 x --threads 2 = 20 cores (matches --cpus-per-task). The
# epsilon levels within one instance are sequential; parallelism is across
# the 56 instances.

module load python/miniconda3-py3.12
source $HOME/movrptw-venv/bin/activate
cd /scratch/lvrmig001/movrptw

python scripts/rerun_gurobi_hard.py 25 \
    --time-limit 1000 \
    --n-jobs 10 \
    --threads 2
    