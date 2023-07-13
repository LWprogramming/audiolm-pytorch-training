#!/bin/bash

# Mostly based on the cocochorales download script. We only care about the wav files so we don't
# need to download the other files related to midi.
# The main dataset is too big for fsx (500+ GB) but in each sample we only need the wav stems,
# not the full mix. After doing this trimming we get around 5.5 GB per folder, so we'll download from 1-60 inclusive = 330 GB to be on the safe side.
# In any case, the audiolm_pytorch code auto-separates code into train/valid/test sets. In reality there are 1-96 for train, and then 1-12 for valid and test each, but we'll just use the first 60 for now.

mkdir /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1 # unzipped
mkdir /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1_zipped
cd /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1_zipped

# download md5 if it doesn't exist already
if [ ! -f "cocochorales_md5s.txt" ]; then
  wget https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/cocochorales_md5s.txt
fi

# download main dataset, specifically train
for i in $(seq 1 1 60); do
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
  aws s3 cp /fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1/"$i" s3://itsleonwu-laion/cocochorales_main_dataset_v1/$i --recursive --profile laion-stability-my-s3-bucket

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

