#!/bin/bash
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=audiolm-e2e
#SBATCH --comment=laion
#SBATCH --output=../audiolm-pytorch-results/output.log
#SBATCH --error=../audiolm-pytorch-results/error.log

python -u audiolm_pytorch_demo_laion.py