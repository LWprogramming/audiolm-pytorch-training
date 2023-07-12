#!/bin/bash

#SBATCH --partition=cpu64
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=download-cocochorales
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --output=../cocochorales-download-results-output-%A.log
#SBATCH --error=../cocochorales-download-results-error-%A.log


# Mostly based on the cocochorales download script. We only care about the wav files so we don't
# need to download the other files related to midi.
# The main dataset is too big for fsx (500+ GB) but in each sample we only need the wav stems,
# not the full mix. In addition, I'm fine with only downloading the training set for now, because
# the audiolm_pytorch code auto-separates code into train/valid/test sets. There look to be
# 96 + 12 + 12  (train + validate + test) = 120 such folders (?) each containing a bunch of samples, and given that the main dataset is 569 GB, I estimate the train alone is around 455ish GB. After saving only the stem_audio wavs in each sample, we reduce the size per-sample to about 3/4 of the original sample folder, so we end up with around a 340ish GB download.
# Just to be on the safe side, we'll split the download into multiple parts and process the unzipping for each part, to ensure no individual segment ends up overloading the fsx storage.

mkdir /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1_zipped
cd /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1_zipped

# download md5
wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/cocochorales_md5s.txt

# download main dataset, specifically train
mkdir main_dataset_train
# TODO: split this into multiple sections; just downloading one for now to proof of concept
#for i in $(seq 1 1 96); do
#  wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/train/"$i".tar.bz2 -P main_dataset_train
#done
wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/train/1.tar.bz2 -P main_dataset_train

# extract the tar files. reqiures pbzip2
python data_download/extract_tars.py --data_dir cocochorales_main_dataset_v1_zipped --output_dir /fsx/itsleonwu/audiolm-pytorch-datasets

