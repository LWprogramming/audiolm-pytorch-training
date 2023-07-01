#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=audiolm-e2e-openslr-dev-clean
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --output=../audiolm-pytorch-results/output-%A.log
#SBATCH --error=../audiolm-pytorch-results/error-%A.log
#SBATCH --requeue

# datetime=$(date +%Y%m%d-%H%M%S)

# example usage: sbatch sbatch_job.sh test_long_sample --with_profiling

echo "SLURM_JOB_ID: $SLURM_JOB_ID" >> ../audiolm-pytorch-results/output-$SLURM_JOB_ID.log
source venv/bin/activate # in case this hasn't already been done

# export CUBLAS_WORKSPACE_CONFIG=:4096:8 # increase memory footprint by about 24 MiB but gives deterministic results. See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
# export CUDA_LAUNCH_BLOCKING=1

RUN_MODE=$1 # required, see audiolm_pytorch_demo_laion.py
WITH_PROFILING=$2 # not required, defaults to no profiling. Usage: --with_profiling
echo "run mode: " $RUN_MODE
python -u audiolm_pytorch_demo_laion.py --slurm_job_id $SLURM_JOB_ID --run_mode $RUN_MODE --parallel_training $WITH_PROFILING

# echo "Model training completed. Now saving results to s3..."
# default to saving to LAION s3 bucket
# python aws_ckpt_backup_script.py $SLURM_JOB_ID
