# audiolm-pytorch-training

This repository contains my scripts to train AudioLM models, using the [audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch) library by lucidrains and the my own modifications in the [personal_hacks branch](https://github.com/LWprogramming/audiolm-pytorch/tree/personal_hacks). The AudioLM model is a mostly 1-1 reproduction of the paper ["AudioLM: a Language Modeling Approach to Audio Generation"](https://arxiv.org/abs/2209.03143).

# Getting started

```bash
# Create necessary directories
mkdir /fsx/itsleonwu
cd /fsx/itsleonwu
mkdir audiolm-pytorch-results
mkdir audiolm-pytorch-datasets

# Clone the audiolm-pytorch-training repository
git clone https://github.com/LWprogramming/audiolm-pytorch-training.git

# Create a virtual environment using Python 3.10 and activate it
cd audiolm-pytorch-training
python3.10 -m venv venv
source venv/bin/activate

# Run the hubert_ckpt_download.py script
python hubert_ckpt_download.py

# Run the use_patched_audiolm.py script
python use_patched_audiolm.py personal_hacks

pip install tensorboardX # for some reason not covered separately

# Download the dataset
cd ../audiolm-pytorch-datasets
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xvf dev-clean.tar.gz
mv LibriSpeech LibriSpeech-dev-clean
rm dev-clean.tar.gz

# Create a directory for the sample file
mkdir many_identical_copies_of_cocochorales_single_sample_resampled_24kHz_trimmed_first_second
echo "Remember to upload the sample file to this overfitting dataset!"
```