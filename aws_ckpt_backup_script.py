import sys
import re
import os
import subprocess

# usage: python aws_ckpt_backup_script.py <job_id> <bucket_prefix optional>
# backs up checkpoints to s3
# my personal bucket is named "itsleonwu-laion"

job_id = sys.argv[1]
if len(sys.argv) == 2:
    bucket_prefix = sys.argv[2]
else:
    bucket_prefix = "s-laion/itsleonwu" # default into my own personal folder 

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
s3_folder = f"slurm-job-{job_id}-audiolm-{audiolm_version}"

# Transfer output and error logs, generation, along with the scripts to run the code.
# I made the bucket back when I was trying paperspace and now it's my catch-all for any temporary ML stuff that I need to back up, oops... naming things is hard
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-training/sbatch_job_{job_id}.sh", f"s3://{bucket_prefix}/{s3_folder}/sbatch_job.sh", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-training/audiolm_pytorch_demo_laion_{job_id}.py", f"s3://{bucket_prefix}/{s3_folder}/audiolm_pytorch_demo_laion.py", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/output-{job_id}.log", f"s3://{bucket_prefix}/{s3_folder}/output-{job_id}.log", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/error-{job_id}.log", f"s3://{bucket_prefix}/{s3_folder}/error-{job_id}.log", "--profile", "laion-stability-my-s3-bucket"])
output_filename = f"out_job_id_{job_id}_step_{step}.wav" # should match what is saved in audiolm_pytorch_demo_laion.py
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/{output_filename}", f"s3://{bucket_prefix}/{s3_folder}/{output_filename}", "--profile", "laion-stability-my-s3-bucket"])


# Transfer checkpoints
for folder in ["semantic_results", "coarse_results", "fine_results"]:
    full_folder_name = f"{folder}_{job_id}"
    folder_path = f"/fsx/itsleonwu/audiolm-pytorch-results/full_folder_name"
    pt_files = [file for file in os.listdir(folder_path) if file.endswith('.pt')]
    max_checkpoint = max(pt_files, key=lambda x: int(re.search(r"(\d+)(?=\.)", x).group(1)))
    subprocess.run(["aws", "s3", "cp", f"{folder_path}/{max_checkpoint}", f"s3://{bucket_prefix}/{s3_folder}/{full_folder_name}/{max_checkpoint}", "--profile", "laion-stability-my-s3-bucket"])