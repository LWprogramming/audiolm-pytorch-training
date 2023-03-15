# assuming running from audiolm-pytorch-training and results in audiolm-pytorch-results

import os
import shutil

results_folder = "../audiolm-pytorch-results"
if not os.path.isdir(results_folder):
	raise AssertionError("didn't find results_folder, no results to clear out")

for folder in [
	"coarse_results",
	"fine_results",
	"semantic_results",
	"soundstream_results",
	"placeholder_dataset",]:
	try:
	    shutil.rmtree(f"{results_folder}/{folder}")
	except FileNotFoundError:
	    pass  # directory does not exist, so do nothing
