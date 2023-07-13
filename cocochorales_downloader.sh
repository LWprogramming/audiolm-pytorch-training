#!/bin/bash

# Mostly based on the cocochorales download script. We only care about the wav files so we don't
# need to download the other files related to midi.
# The main dataset is too big for fsx (500+ GB) but in each sample we only need the wav stems,
# not the full mix. In addition, I'm fine with only downloading the training set for now, because
# the audiolm_pytorch code auto-separates code into train/valid/test sets. There look to be
# 96 + 12 + 12  (train + validate + test) = 120 such folders (?) each containing a bunch of samples, and given that the main dataset is 569 GB, I estimate the train alone is around 455ish GB. After saving only the stem_audio wavs in each sample, we reduce the size per-sample to about 3/4 of the original sample folder, so we end up with around a 340ish GB download.
# Just to be on the safe side, we'll split the download into multiple parts and process the unzipping for each part, to ensure no individual segment ends up overloading the fsx storage.

mkdir /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1 # unzipped
mkdir /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1_zipped
cd /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1_zipped

# download md5
wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/cocochorales_md5s.txt

# download main dataset, specifically train
for i in $(seq 1 1 96); do
  # wget if $i.tar.bz2 doesn't exist in cocochorales_main_dataset_v1_zipped
  if [ ! -f "$i".tar.bz2 ]; then
    wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/train/"$i".tar.bz2
  fi
  # copy to s3, zipped. only run this once!
  # aws s3 cp /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1_zipped/"$i".tar.bz2 s3://s-laion/itsleonwu-laion/cocochorales_main_dataset_v1_zipped --profile laion-stability-my-s3-bucket

  cd ../cocochorales_main_dataset_v1
  mkdir $i
  tar -xjf ../cocochorales_main_dataset_v1_zipped/$i.tar.bz2 -C ./$i
  # copy to s3, unzipped. only run this once!
  # TODO: if i ever uncomment this: need to add dir "$i" to s3 path
  # aws s3 cp /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1/"$i" s3://itsleonwu-laion/cocochorales_main_dataset_v1/ --recursive --profile laion-stability-my-s3-bucket

  # Cleanup: keep only stem wavs
  cd $i
  # Loop through subfolders
  for subfolder in */; do
    cd "$subfolder"
    cp stems_audio/*.wav . # copy to subfolder
    # Delete all other files
    rm -rf stems_audio stems_midi metadata.yaml mix.mid mix.wav
    cd .. # back to $i
  done
  cd .. # back to cocochorales_main_dataset_v1
  echo "completed $i"
  cd ../cocochorales_main_dataset_v1_zipped
done

