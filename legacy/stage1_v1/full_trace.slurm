#!/bin/bash
#SBATCH --job-name=s1-full
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/full-%j.out
#SBATCH --error=runs/full-%j.err

echo "=========================================="
echo "SLURM_JOB_ID   = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="

CONDA_ENV="${CONDA_ENV:-base}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

REPO="$HOME/ANLP-Group-Project"
cd "$REPO"

mkdir -p runs

python stage1/full_trace.py

echo "Exit code: $?"