import sys
import re
import os
import subprocess

job_id = sys.argv[1]

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
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-training/sbatch_job.sh", f"s3://itsleonwu-paperspace/{s3_folder}/sbatch_job.sh", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-training/audiolm_pytorch_demo_laion.py", f"s3://itsleonwu-paperspace/{s3_folder}/audiolm_pytorch_demo_laion.py", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/output-{job_id}.log", f"s3://itsleonwu-paperspace/{s3_folder}/output-{job_id}.log", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/error-{job_id}.log", f"s3://itsleonwu-paperspace/{s3_folder}/error-{job_id}.log", "--profile", "laion-stability-my-s3-bucket"])
subprocess.run(["aws", "s3", "cp", f"/fsx/itsleonwu/audiolm-pytorch-results/out_{job_id}.wav", f"s3://itsleonwu-paperspace/{s3_folder}/error-{job_id}.log", "--profile", "laion-stability-my-s3-bucket"])


# Transfer checkpoints
for folder in ["semantic_results", "coarse_results", "fine_results"]:
    folder_path = f"/fsx/itsleonwu/audiolm-pytorch-results/{folder}_{job_id}"
    pt_files = [file for file in os.listdir(folder_path) if file.endswith('.pt')]
    max_checkpoint = max(pt_files, key=lambda x: int(re.search(r"(\d+)(?=\.)", x).group(1)))
    subprocess.run(["aws", "s3", "cp", f"{folder_path}/{max_checkpoint}", f"s3://itsleonwu-paperspace/{s3_folder}/{folder}/{max_checkpoint}", "--profile", "laion-stability-my-s3-bucket"])