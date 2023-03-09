import os
import urllib

prefix = "/fsx/itsleonwu/audiolm-pytorch"
dataset_folder = f"{prefix}/placeholder_dataset"
hubert_ckpt = f'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer

# hubert checkpoints can be downloaded at
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
if not os.path.isdir("hubert"):
  os.makedirs("hubert")
if not os.path.isfile(f"{prefix}/{hubert_ckpt}"):
  hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
  urllib.request.urlretrieve(hubert_ckpt_download, f"{prefix}/{hubert_ckpt}")
if not os.path.isfile(f"{prefix}/{hubert_quantizer}"):
  hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
  urllib.request.urlretrieve(hubert_quantizer_download, f"{prefix}/{hubert_quantizer}")
