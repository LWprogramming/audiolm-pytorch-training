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

# Sometimes slurm jobs get pre-empted. If this ends up happening, we want to have two things recorded: the current training script, so we can properly restart training (in case there were breaking changes previously made). It'd take a good bit more effort to save a separate audiolm_pytorch version for each run, while the API for that doesn't change much, so I'm going to skip that for now.
# See also https://twitter.com/miraculous_cake/status/1676003814372151297
# This is somewhat like the fork() function in unix operating systems
if [[ -f audiolm_pytorch_demo_laion_$SLURM_JOB_ID.py && -f sbatch_script_$SLURM_JOB_ID.sh ]]; then
    echo "SLURM_JOB_ID: $SLURM_JOB_ID" >> ../audiolm-pytorch-results/output-$SLURM_JOB_ID.log
	source venv/bin/activate # in case this hasn't already been done
	# export CUBLAS_WORKSPACE_CONFIG=:4096:8 # increase memory footprint by about 24 MiB but gives deterministic results. See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
	# export CUDA_LAUNCH_BLOCKING=1
	RUN_MODE=$1 # required, see audiolm_pytorch_demo_laion.py
	WITH_PROFILING=$2 # not required, defaults to no profiling. Usage: --with_profiling
	echo "run mode: " $RUN_MODE
	python -u audiolm_pytorch_demo_laion.py --slurm_job_id $SLURM_JOB_ID --run_mode $RUN_MODE --parallel_training $WITH_PROFILING
else
    cp audiolm_pytorch_demo_laion.py audiolm_pytorch_demo_laion_$SLURM_JOB_ID.py
    cp sbatch_script.sh sbatch_script_$SLURM_JOB_ID.sh
	sbatch sbatch_script_$SLURM_JOB_ID.sh "$@"
fi
