# assuming running from audiolm-pytorch-training and results in audiolm-pytorch-results

import os
import shutil
import re
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Clear out specific training job results.')
parser.add_argument('job_ids', type=int, nargs='+', help='List of training job ids to clear.')
args = parser.parse_args()

# Confirm deletion
x = input("Are you SURE you want to wipe out previous results? Type \"absolutely yes\" if so")
if not x == "absolutely yes":
    raise AssertionError("nope")

results_folder = "../audiolm-pytorch-results"
if not os.path.isdir(results_folder):
    raise AssertionError("didn't find results_folder, no results to clear out")

prefixes = [
    "coarse_results",
    "fine_results",
    "semantic_results",
]

for job_id in args.job_ids:
    for item in os.listdir(results_folder):
        item_path = os.path.join(results_folder, item)
        for prefix in prefixes:
            if item.startswith(f"{prefix}_{job_id}"):
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                break
        if re.match(rf'(output|error)-{job_id}\.log', item):
            if os.path.isfile(item_path):
                os.remove(item_path)

# Usage: python script.py [job_id1] [job_id2] ... [job_idN]
# Example: python script.py 123 456 789
