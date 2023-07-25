#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=audiolm-e2e-openslr-dev-clean
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --output=../audiolm-pytorch-results/output-%A.log
#SBATCH --error=../audiolm-pytorch-results/error-%A.log
#SBATCH --requeue

# datetime=$(date +%Y%m%d-%H%M%S)

# example usage: sbatch sbatch_job.sh -r test_long_sample -p --with_profiling -s 123456

# parse arguments
# Unfortunately, the `getopts` function in bash only supports single-character options, can't name it without a more complicated solution
while getopts "r:p:s:c:t:" opt; do
  case ${opt} in
    r)
      RUN_MODE=$OPTARG
      ;;
    p)
      WITH_PROFILING=$OPTARG
      ;;
    s)
      POTENTIAL_ALTERNATE_SLURM_JOB_ID=$OPTARG
      ;;
    c)
      CHECKPOINT_SLURM_JOB_ID=$OPTARG
      ;;
    t)
      TRANSFORMER_TO_TARGET=$OPTARG # should be one of semantic, coarse, or fine. or just leave it blank for eval mode.
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# OVERRIDABLE_SLURM_JOB_ID decides whether to use an old slurm job's training script
OVERRIDABLE_SLURM_JOB_ID=${POTENTIAL_ALTERNATE_SLURM_JOB_ID:-$SLURM_JOB_ID}  # use this job's slurm job id by default, but allow overriding it with a custom value
# CHECKPOINT_SLURM_JOB_ID decides whether to use an old slurm job's checkpoint
CHECKPOINT_SLURM_JOB_ID=${CHECKPOINT_SLURM_JOB_ID:-$SLURM_JOB_ID}

# Sometimes slurm jobs get pre-empted. If this ends up happening, we want to have two things recorded: the current training script, so we can properly restart training (in case there were breaking changes previously made). It'd take a good bit more effort to save a separate audiolm_pytorch version for each run, while the API for that doesn't change much, so I'm going to skip that for now.
# See also https://twitter.com/miraculous_cake/status/1676003814372151297
# This is somewhat like the fork() function in unix operating systems
if [[ ! -f audiolm_pytorch_demo_laion_$OVERRIDABLE_SLURM_JOB_ID.py || ! -f sbatch_job_$OVERRIDABLE_SLURM_JOB_ID.sh ]]; then
  cp audiolm_pytorch_demo_laion.py audiolm_pytorch_demo_laion_$OVERRIDABLE_SLURM_JOB_ID.py
  cp sbatch_job.sh sbatch_job_$OVERRIDABLE_SLURM_JOB_ID.sh
fi

echo "SLURM_JOB_ID: $OVERRIDABLE_SLURM_JOB_ID" >> ../audiolm-pytorch-results/output-$OVERRIDABLE_SLURM_JOB_ID.log
source venv/bin/activate # in case this hasn't already been done
# export CUBLAS_WORKSPACE_CONFIG=:4096:8 # increase memory footprint by about 24 MiB but gives deterministic results. See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
# export CUDA_LAUNCH_BLOCKING=1

echo "run mode: " $RUN_MODE
echo "with profiling: " $WITH_PROFILING
echo "slurm job id to actually use: " $OVERRIDABLE_SLURM_JOB_ID

# Transformers need to be trained separately, see: https://github.com/lucidrains/audiolm-pytorch/issues/209#issuecomment-1640777646
# Set default to eval mode
TRAIN_OR_EVAL=""
# Check if transformer target is set
if [ -n "$TRANSFORMER_TO_TARGET" ]; then
  TRAIN_OR_EVAL="train_$TRANSFORMER_TO_TARGET"
  accelerate launch audiolm_pytorch_demo_laion_$OVERRIDABLE_SLURM_JOB_ID.py --run_mode $RUN_MODE $WITH_PROFILING --slurm_job_id $CHECKPOINT_SLURM_JOB_ID --train_or_eval $TRAIN_OR_EVAL
else
  TRAIN_OR_EVAL="evaluate"
  python audiolm_pytorch_demo_laion_$OVERRIDABLE_SLURM_JOB_ID.py --run_mode $RUN_MODE $WITH_PROFILING --slurm_job_id $CHECKPOINT_SLURM_JOB_ID --train_or_eval $TRAIN_OR_EVAL
fi
