#!/bin/bash
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=audiolm-e2e
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --output=../audiolm-pytorch-results/output-%A.log
#SBATCH --error=../audiolm-pytorch-results/error-%A.log
#SBATCH --requeue

# datetime=$(date +%Y%m%d-%H%M%S)

echo "SLURM_JOB_ID: $SLURM_JOB_ID" >> ../audiolm-pytorch-results/output-$SLURM_JOB_ID.log

export CUBLAS_WORKSPACE_CONFIG=:4096:8 # increase memory footprint by about 24 MiB but gives deterministic results. See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

python -u audiolm_pytorch_demo_laion.py "$@" --slurm_job_id $SLURM_JOB_ID