import sys
import re
import os
import subprocess

# usage: python aws_ckpt_backup_script.py <job_id> <checkpoint_job_id> <bucket_prefix optional>

# aws cp commands in shell look like aws s3 cp /fsx/itsleonwu/audiolm-pytorch-training/sbatch_job_1.sh s3://s-laion/itsleonwu/slurm-job-1-audiolm-0.1.0/sbatch_job.sh --profile laion-stability-my-s3-bucket

# backs up checkpoints to s3
# my personal bucket is named "itsleonwu-laion"

# job id is the job that generated the script, checkpoint_job_id is the id of the folders with checkpoints. If these are the same, they should appear twice. They differ when you start a new job that continues from an old folder's checkpoint
job_id = sys.argv[1]
checkpoint_job_id = sys.argv[2]
if len(sys.argv) >= 4:
    bucket_prefix = sys.argv[3]
else:
    bucket_prefix = "s-laion/itsleonwu"  # default into my own personal folder

description = input("Write a description or leave it blank if you don't want to:")
with open(f"/fsx/itsleonwu/audiolm-pytorch-results/{job_id}-description.txt", "w") as f:
    f.write(description)

# Read output log
with open(f"/fsx/itsleonwu/audiolm-pytorch-results/output-{job_id}.log", "r") as f:
    output_log = f.read()

# Extract audiolm version
version_match = re.search(r"training on audiolm_pytorch version (\d+\.\d+\.\d+)", output_log)
audiolm_version = version_match.group(1)

# Define S3 bucket folder
s3_scripts_folder = f"slurm-job-{job_id}-audiolm-{audiolm_version}"
s3_checkpoints_folder = f"slurm-job-{checkpoint_job_id}-audiolm-{audiolm_version}"

# Transfer output and error logs, generation, along with the scripts to run the code.
# I made the bucket back when I was trying paperspace and now it's my catch-all for any temporary ML stuff that I need to back up, oops... naming things is hard
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-training/sbatch_job_{job_id}.sh", f"s3://{bucket_prefix}/{s3_scripts_folder}/sbatch_job.sh", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-training/audiolm_pytorch_demo_laion_{job_id}.py", f"s3://{bucket_prefix}/{s3_scripts_folder}/audiolm_pytorch_demo_laion.py", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/output-{job_id}.log", f"s3://{bucket_prefix}/{s3_scripts_folder}/output-{job_id}.log", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/error-{job_id}.log", f"s3://{bucket_prefix}/{s3_scripts_folder}/error-{job_id}.log", "--profile", "laion-stability-my-s3-bucket"])
# output_filename = f"out_job_id_{job_id}_step_{step}.wav" # should match what is saved in audiolm_pytorch_demo_laion.py
# subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/{output_filename}", f"s3://{bucket_prefix}/{s3_folder}/{output_filename}", "--profile", "laion-stability-my-s3-bucket"])


# Transfer checkpoints
for folder in ["semantic_results", "coarse_results", "fine_results"]:
    full_folder_name = f"{folder}_{checkpoint_job_id}"
    folder_path = f"/fsx/itsleonwu/audiolm-pytorch-results/{full_folder_name}"
    if os.path.exists(folder_path):
        # probably only one of these will exist if we're training different transformers in different jobs
        pt_files = [file for file in os.listdir(folder_path) if file.endswith('.pt')]
        if pt_files:
            max_checkpoint = max(pt_files, key=lambda x: int(re.search(r"(\d+)(?=\.)", x).group(1)))
            subprocess.run(["aws", "s3", "cp", f"{folder_path}/{max_checkpoint}", f"s3://{bucket_prefix}/{s3_checkpoints_folder}/{full_folder_name}/{max_checkpoint}", "--profile", "laion-stability-my-s3-bucket"])