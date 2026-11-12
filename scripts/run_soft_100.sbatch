#!/bin/bash
#SBATCH --job-name=movrptw_soft_100
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/lvrmig001/movrptw/logs/soft100-%j.out
#SBATCH --error=/scratch/lvrmig001/movrptw/logs/soft100-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lvrmig001@myuct.ac.za

source $HOME/movrptw-venv/bin/activate
cd /scratch/lvrmig001/movrptw

python scripts/run_experiment.py \
    --config configs/experiment_soft_100.yaml \
    --logfile logs/soft100-$SLURM_JOB_ID.log