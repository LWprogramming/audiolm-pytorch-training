# assuming running from audiolm-pytorch-training and results in audiolm-pytorch-results

import os
import shutil

results_folder = "../audiolm-pytorch-results"
if not os.path.isdir(results_folder):
	raise AssertionError("didn't find results_folder, no results to clear out")

shutil.rmtree(f"{results_folder}/coarse_results")
shutil.rmtree(f"{results_folder}/fine_results")
shutil.rmtree(f"{results_folder}/semantic_results")
shutil.rmtree(f"{results_folder}/soundstream_results")
shutil.rmtree(f"{results_folder}/placeholder_dataset")