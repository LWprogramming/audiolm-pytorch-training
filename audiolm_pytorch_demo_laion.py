# imports
import math
import wave
import struct
import os
import urllib.request
# import tarfile
import audiolm_pytorch
from audiolm_pytorch import AudioLMSoundStream, SoundStreamTrainer
from audiolm_pytorch import EncodecWrapper
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from torch import nn
import torch
import torchaudio
from torch.profiler import profile, record_function, ProfilerActivity
import datetime
import argparse
import re

import random
import numpy as np
from torch.utils.data import DataLoader

# import boto3
# import datetime
# from botocore.errorfactory import ClientError

# Eliminate ALL non-determinism https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True) # doesn't work due to https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/10

# For logging
print(f"PyTorch seed: {torch.initial_seed()}")
print(f"NumPy seed: {np.random.get_state()[1][0]}")
# print(f"Random seed: {random.getstate()[1][0]}") # never mind, random seed is not accessible in python
torch.backends.cudnn.benchmark = False

# Checkpoint loading. Expect format to be something like /path/to/semantic.transformer.20000.pt
parser = argparse.ArgumentParser()
parser.add_argument('--slurm_job_id', type=int, help='slurm job id, used for creating results folders', required=True)
# parallel vs non-parallel training
parser.add_argument('--no_parallel_training', dest='parallel_training', help="disable parallel training, forcing transformers to train in sequence.", action='store_false')
parser.add_argument('--parallel_training', dest='parallel_training', help="enable parallel training, forcing transformers to train a little bit before handing off to the next transformer. Good for seeing how results progressively get better.", action='store_true')
parser.add_argument('--run_mode',
                    type=str,
                    help='run configuration (pick from choices). Sets dataset_folder, num_train_steps, save_every, batch_size, and grad_accum_every',
                    choices=["openslr", "cocochorales_overfit_1_second", "cocochorales_overfit", "cocochorales_test_custom_dataset"],
                    default=None,
                    required=True)
parser.set_defaults(parallel_training=False)
args = parser.parse_args()
results_folder_suffix = str(args.slurm_job_id)
print("parsed args")

 # dataset and dataset_folder generally don't show up together-- only one will be defined per run configuration
if args.run_mode == "openslr":
    dataset = None
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/LibriSpeech-dev-clean/dev-clean"
    num_train_steps = 1000001
    save_every = 5000
    batch_size = 8
    grad_accum_every = 16
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "cocochorales_overfit_1_second":
    # resample the given sample to 24kHz to work with encodec and then trim it so we take only the first second of audio, so the transformer actually only sees the same data every single time
    dataset = None
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/many_identical_copies_of_cocochorales_single_sample_resampled_24kHz_trimmed_first_second"
    num_train_steps = 5001
    save_every = 1000
    batch_size = 1
    grad_accum_every = 1
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "cocochorales_overfit":
    # try a single un-trimmed data point direct from cocochorales, default at 16kHz
    dataset = None
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_single_sample_unprocessed"
    num_train_steps = 5001
    save_every = 1000
    batch_size = 1
    grad_accum_every = 1
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "cocochorales_test_custom_dataset":
    # try writing a custom Dataset for concatenating samples to learn accompaniments
    raise AssertionError("not implemented yet")
    dataset = None
    dataset_folder = None
    num_train_steps = 5001
    save_every = 1000
    batch_size = 1
    grad_accum_every = 1
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "test_long_sample":
    # try out really long lengths to get a sense of how long the data input can be
    dataset = None
    # data generated from mix.wav with shell command
    # for i in {1..30}; do ffmpeg -i mix.wav -ss 3 -to 17 -c copy segment$i.wav; done && for i in {1..30}; do printf "file '%s'\n" segment$i.wav >> list.txt; done && ffmpeg -f concat -safe 0 -i list.txt -c copy output.wav && rm segment*.wav list.txt
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/test_long_sample"
    num_train_steps = 101
    save_every = 50
    batch_size = 1
    grad_accum_every = 1
    data_max_length = None
    data_max_length_seconds = 5
else:
    raise AssertionError("impossible to be here, bug")

# Usage:
# python audiolm_pytorch_demo_laion.py --semantic=/path/to/semantic --coarse=/path/to/coarse --fine=/path/to/fine
# Checkpoint flags are optional of course. You need to give a full path, no guarantees if it's not a full path.
# define all dataset paths, checkpoints, etc
prefix = "/fsx/itsleonwu/audiolm-pytorch-results"
hubert_ckpt = f'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer

print(f"training on audiolm_pytorch version {audiolm_pytorch.version.__version__}")
def get_potential_checkpoint_path(transformer_name, trainer, prefix, results_folder_suffix):
    """Determine checkpoint paths based on slurm_job_id CLI argument. searches in `prefix` folder) or latest available checkpoints in `prefix` folder. Returns None if no such checkpoints exist at all."""
    assert transformer_name in {"semantic", "coarse", "fine"}

    results_folder = f"{prefix}/{transformer_name}_results_{results_folder_suffix}"
    if not os.path.exists(results_folder):
        return None

    checkpoints = [f for f in os.listdir(results_folder) if f.endswith('.pt')]
    steps = [int(re.findall(r'\d+', ckpt)[-1]) for ckpt in checkpoints]
    max_step = max(steps, default=0)

    if max_step % trainer.save_model_every != 0 or max_step > trainer.num_train_steps:
        raise ValueError("Invalid checkpoint step")

    return f"{results_folder}/{transformer_name}.transformer.{max_step}.pt" if max_step > 0 else None


# Placeholder data generation
def get_sinewave(freq=440.0, duration_ms=200, volume=1.0, sample_rate=24000.0):
  # code adapted from https://stackoverflow.com/a/33913403
  audio = []
  num_samples = duration_ms * (sample_rate / 1000.0)
  for x in range(int(num_samples)):
    audio.append(volume * math.sin(2 * math.pi * freq * (x / sample_rate)))
  return audio

def save_wav(file_name, audio, sample_rate=24000.0):
  # Open up a wav file
  wav_file=wave.open(file_name,"w")
  # wav params
  nchannels = 1
  sampwidth = 2
  # 24000 is the industry standard sample rate - CD quality.  If you need to
  # save on file size you can adjust it downwards. The stanard for low quality
  # is 8000 or 8kHz.
  nframes = len(audio)
  comptype = "NONE"
  compname = "not compressed"
  wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))
  # WAV files here are using short, 16 bit, signed integers for the 
  # sample size.  So we multiply the floating point data we have by 32767, the
  # maximum value for a short integer.  NOTE: It is theortically possible to
  # use the floating point -1.0 to 1.0 data directly in a WAV file but not
  # obvious how to do that using the wave module in python.
  for sample in audio:
      wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))
  wav_file.close()
  return

def make_placeholder_dataset():
  # Make a placeholder dataset with a few .wav files that you can "train" on, just to verify things work e2e
  if os.path.isdir(dataset_folder):
    return
  os.makedirs(dataset_folder)
  save_wav(f"{dataset_folder}/example.wav", get_sinewave(duration_ms=1000))
  save_wav(f"{dataset_folder}/example2.wav", get_sinewave(duration_ms=1050))
  save_wav(f"{dataset_folder}/example3.wav", get_sinewave(duration_ms=1002, freq=515.0))
  save_wav(f"{dataset_folder}/example4.wav", get_sinewave(duration_ms=1003, freq=334.0))
  os.makedirs(f"{dataset_folder}/subdirectory")
  save_wav(f"{dataset_folder}/subdirectory/example.wav", get_sinewave(freq=330.0))
# make_placeholder_dataset()

#######

# codec = AudioLMSoundStream(
#     codebook_size = 1024,
#     rq_num_quantizers = 8,
#     attn_window_size = 128,       # local attention receptive field at bottleneck
#     attn_depth = 2                # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
# )

# soundstream_trainer = SoundStreamTrainer(
#     codec,
#     folder = dataset_folder,
#     lr=3e-4,
#     batch_size = 4,
#     grad_accum_every = 8, # effective batch size of batch_size * grad_accum_every = 32
#     data_max_length_seconds = 2,  # train on 2 second audio
#     results_folder = f"{prefix}/soundstream_results_{results_folder_suffix}",
#     save_results_every = 4,
#     save_model_every = 4,
#     num_train_steps = 9
# ).cuda()

# soundstream_trainer.train()

codec = EncodecWrapper()

#############

wav2vec = HubertWithKmeans(
    # use_mert = True,
    checkpoint_path = f"{prefix}/{hubert_ckpt}",
    # checkpoint_path = None,
    kmeans_path = f"{prefix}/{hubert_quantizer}"
)

num_train_steps = 1000001
save_every = 100000

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()

semantic_trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    dataset = dataset, # dataset and folder generally don't show up together-- only one will be defined per run configuration
    folder = dataset_folder,
    batch_size = 8,
    grad_accum_every = 16,
    data_max_length = data_max_length,
    data_max_length_seconds = data_max_length_seconds,
    num_train_steps = num_train_steps,
    save_results_every = save_every,
    save_model_every = save_every,
    results_folder = f"{prefix}/semantic_results_{results_folder_suffix}",
    force_clear_prev_results = False,
)

semantic_ckpt = get_potential_checkpoint_path("semantic", semantic_trainer, prefix, results_folder_suffix)
print(f"loading semantic checkpoint {semantic_ckpt}")
if semantic_ckpt is not None:
    semantic_trainer.load(semantic_ckpt)

# semantic_trainer.train()

################

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

coarse_trainer = CoarseTransformerTrainer(
    transformer = coarse_transformer,
    codec = codec,
    wav2vec = wav2vec,
    dataset = dataset, # dataset and folder generally don't show up together-- only one will be defined per run configuration
    folder = dataset_folder,
    batch_size = 8,
    grad_accum_every = 16,
    data_max_length = data_max_length,
    data_max_length_seconds = data_max_length_seconds,
    results_folder = f"{prefix}/coarse_results_{results_folder_suffix}",
    num_train_steps = num_train_steps,
    save_results_every = save_every,
    save_model_every = save_every,
    force_clear_prev_results = False,
)

coarse_ckpt = get_potential_checkpoint_path("coarse", coarse_trainer, prefix, results_folder_suffix)
print(f"loading coarse checkpoint {coarse_ckpt}")
if coarse_ckpt is not None:
    coarse_trainer.load(coarse_ckpt)

# coarse_trainer.train()

################

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

fine_trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    codec = codec,
    dataset = dataset, # dataset and folder generally don't show up together-- only one will be defined per run configuration
    folder = dataset_folder,
    batch_size = 8,
    grad_accum_every = 16,
    data_max_length = data_max_length,
    data_max_length_seconds = data_max_length_seconds,
    num_train_steps = num_train_steps,
    save_results_every = save_every,
    save_model_every = save_every,
    results_folder = f"{prefix}/fine_results_{results_folder_suffix}",
    force_clear_prev_results = False,
)

fine_ckpt = get_potential_checkpoint_path("fine", fine_trainer, prefix, results_folder_suffix)
print(f"loading fine checkpoint {fine_ckpt}")
if fine_ckpt is not None:
    fine_trainer.load(fine_ckpt)

# fine_trainer.train()

################
# All together now
def get_sample(wav2vec, codec, semantic_transformer, coarse_transformer, fine_transformer, step):
    # Generate output and save
    audiolm = AudioLM(
        wav2vec = wav2vec,
        codec = codec,
        semantic_transformer = semantic_transformer,
        coarse_transformer = coarse_transformer,
        fine_transformer = fine_transformer
    )

    generated_wav = audiolm(batch_size = 1)
    output_path = f"{prefix}/out_job_id_{args.slurm_job_id}_step_{step}.wav"
    sample_rate = 24000
    torchaudio.save(output_path, generated_wav.cpu(), sample_rate)

if args.parallel_training:
    print("training in parallel")
    def train_models(steps_to_train):
        for _ in range(steps_to_train):
            semantic_trainer.train_step()
        for _ in range(steps_to_train):
            coarse_trainer.train_step()
        for _ in range(steps_to_train):
            fine_trainer.train_step()

    for step in range(0, num_train_steps, save_every):
        train_models(save_every)
        get_sample(wav2vec, codec, semantic_transformer, coarse_transformer, fine_transformer, step)
else:
    # non parallel training
    print("not training in parallel")
    semantic_trainer.train()
    coarse_trainer.train()
    fine_trainer.train()
    get_sample(wav2vec, codec, semantic_transformer, coarse_transformer, fine_transformer, num_train_steps)


# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
#     with record_function("model_inference"):
#         generated_wav = audiolm(batch_size = 1)
#         output_path = f"{prefix}/out.wav"
#         sample_rate = 24000
#         torchaudio.save(output_path, generated_wav.cpu(), sample_rate)

# filename = f"{prefix}/profile-{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.txt"
# with open(filename, "w") as f:
#     f.write("cpu time sorted:\n")
#     f.write(f"{prof.key_averages(group_by_input_shape=True).table(sort_by='cpu_time_total', row_limit=10)}")
#     f.write("\n cuda time sorted:\n")
#     f.write(f"{prof.key_averages().table(sort_by='cuda_time_total', row_limit=10)}")
#     f.write("\ncpu memory self\n") # excludes children memory allocated
#     f.write(f"{prof.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=10)}")
#     f.write("\ncpu memory\n") # includes children memory allocated
#     f.write(f"{prof.key_averages().table(sort_by='cpu_memory_usage', row_limit=10)}\n")






