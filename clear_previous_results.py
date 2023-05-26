# assuming running from audiolm-pytorch-training and results in audiolm-pytorch-results

import os
import shutil

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
    "soundstream_results",
    "placeholder_dataset",
]

for item in os.listdir(results_folder):
    item_path = os.path.join(results_folder, item)
    for prefix in prefixes:
        if item.startswith(prefix):
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
            break