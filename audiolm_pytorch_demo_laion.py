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
from torch.profiler import profile, record_function, ProfilerActivity, schedule
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
parser.add_argument('--train_or_eval', type=str, help="decide which transformer to train (pick from choices)", choices=["train_semantic", "train_coarse", "train_fine", "evaluate"], required=True)
parser.add_argument('--run_mode',
                    type=str,
                    help='run configuration (pick from choices). Sets dataset_folder, num_train_steps, save_every, batch_size, and grad_accum_every',
                    choices=["openslr",
                            "bare_minimum",
                            "cocochorales_overfit_1_second",
                            "cocochorales_overfit",
                            "cocochorales_test_custom_dataset",
                            "test_long_sample"],
                    default=None,
                    required=True)
parser.add_argument("--with_profiling", type=bool, default=False, nargs='?', const=True)
args = parser.parse_args()
if args.with_profiling:
    raise NotImplementedError("Profiling is not implemented yet. see train() function below")
results_folder_suffix = str(args.slurm_job_id)
print("parsed args")

# dataset and dataset_folder generally don't show up together-- only one will be defined per run configuration
if args.run_mode == "openslr":
    dataset = None
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/LibriSpeech-dev-clean/dev-clean"
    num_train_steps = 1000000
    save_every = 5000
    batch_size = 8
    grad_accum_every = 16
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "bare_minimum":
    dataset = None
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/many_identical_copies_of_cocochorales_single_sample_resampled_24kHz_trimmed_first_second"
    num_train_steps = 10 # i guess need for avoid off by one error
    save_every = 3
    batch_size = 1
    grad_accum_every = 1
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "cocochorales_overfit_1_second":
    # resample the given sample to 24kHz to work with encodec and then trim it so we take only the first second of audio, so the transformer actually only sees the same data every single time
    dataset = None
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/many_identical_copies_of_cocochorales_single_sample_resampled_24kHz_trimmed_first_second"
    num_train_steps = 501
    save_every = 100
    batch_size = 1
    grad_accum_every = 1
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "cocochorales_overfit":
    # try a single un-trimmed data point direct from cocochorales, default at 16kHz
    dataset = None
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_single_sample_unprocessed"
    num_train_steps = 5000
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
    num_train_steps = 5000
    save_every = 1000
    batch_size = 1
    grad_accum_every = 1
    data_max_length = 24000
    data_max_length_seconds = None
elif args.run_mode == "test_long_sample":
    # try out really long lengths to get a sense of how long the data input can be
    # Measurements on a single NVIDIA A100-SXM4-40GB:
    # 20 train steps on this took around 3:40 = 220 seconds, so 1 step in ~10 seconds (if you account for startup taking some time)
    # Memory usage: 9778 / 40960MiB
    dataset = None
    # data generated from mix.wav with shell command
    # for i in {1..30}; do ffmpeg -i mix.wav -ss 3 -to 17 -c copy segment$i.wav; done && for i in {1..30}; do printf "file '%s'\n" segment$i.wav >> list.txt; done && ffmpeg -f concat -safe 0 -i list.txt -c copy output.wav && rm segment*.wav list.txt
    # Then in /fsx, copy it so you have enough for batch up to 32
    # for i in {0..32}; do cp output.wav output_copy_$i.wav; done
    dataset_folder = "/fsx/itsleonwu/audiolm-pytorch-datasets/test_long_sample"
    num_train_steps = 100
    save_every = 50
    batch_size = 8
    grad_accum_every = 16
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
        raise ValueError(f"Invalid checkpoint step, with max_step {max_step} and save_model_every {trainer.save_model_every} and num_train_steps {trainer.num_train_steps}")

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
    batch_size = batch_size,
    grad_accum_every = grad_accum_every,
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
    batch_size = batch_size,
    grad_accum_every = grad_accum_every,
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
    batch_size = batch_size,
    grad_accum_every = grad_accum_every,
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

def train(profiler=None):
    # pass in profiler as an optional argument. If we're not doing profiling, then nothing happens.
    # TODO: this code needs to happen SOMEWHERE if we're doing profiling.
    # TODO: see: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs for more info on profiler.step() and schedules. We use the second save_every steps instead of the first in case there's any weird overhead to setting things up.
    # if profiler is not None:
    #     profiler.step()
    print(f"hi. args train or eval is {args.train_or_eval}")
    if args.train_or_eval == "evaluate":
        print(f"entered evaluate branch")
        step = semantic_trainer.steps.item()
        print(f"semantic steps {semantic_trainer.steps.item()}")
        # assert semantic_trainer.steps.item() == coarse_trainer.steps.item() and coarse_trainer.steps.item() == fine_trainer.steps.item(), "all three trainers should have the same number of steps when fully trained"
        print(f"coarse trainer device {coarse_trainer.device}")
        print(f"semantic trainer is main {semantic_trainer.is_main}")
        if semantic_trainer.is_main:
            print(f"corase trainer device {coarse_trainer.device}")
            get_sample(wav2vec, codec, semantic_transformer, coarse_transformer, fine_transformer, step)
        return
    elif args.train_or_eval == "train_semantic":
        trainer = semantic_trainer
    elif args.train_or_eval == "train_coarse":
        trainer = coarse_trainer
    elif args.train_or_eval == "train_fine":
        trainer = fine_trainer
    else:
        raise AssertionError(f"train_or_eval argument {args.train_or_eval} not recognized, should be unreachable")
    print(f"training using {args.train_or_eval} trainer")
    trainer.train()

def trace_handler(prof):
    profile_log = f"{prefix}/profiler_{args.slurm_job_id}.txt"
    # Note the difference between self cpu time and cpu time - operators can call other operators, self cpu time excludes time spent in children operator calls, while total cpu time includes it.
    print(f"arrived in trace_handler, logfile name {profile_log}")
    with open(profile_log, "w") as f:
        f.write("cpu_time_total:\n")
        f.write(f"{prof.key_averages(group_by_input_shape=True).table(sort_by='cpu_time_total', row_limit=10)}")
        f.write("\nself_cpu_time_total:\n") 
        f.write(f"{prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10)}")
        f.write("\ncpu_memory_usage:\n")
        f.write(f"{prof.key_averages().table(sort_by='cpu_memory_usage', row_limit=10)}\n")
        f.write("\nself_cpu_memory_usage:\n") 
        f.write(f"{prof.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=10)}")

        f.write("cuda_time_total:\n")
        f.write(f"{prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total', row_limit=10)}")
        f.write("\nself_cuda_time_total:\n") 
        f.write(f"{prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10)}")
        f.write("\ncuda_memory_usage:\n")
        f.write(f"{prof.key_averages().table(sort_by='cuda_memory_usage', row_limit=10)}\n")
        f.write("\nself_cuda_memory_usage:\n") 
        f.write(f"{prof.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=10)}")

    # Also try this:
    # You can examine the sequence of profiled operators and CUDA kernels in Chrome trace viewer (chrome://tracing):
    prof.export_chrome_trace(f"{prefix}/trace_{args.slurm_job_id}.json")

    # export stacks to check out flamegraph too
    # TODO: if this works, try it
    # prof.export_stacks("/tmp/profiler_stacks_{args.slurm_job_id}.txt", "self_cuda_time_total")
    # From the profiler docs:
    # We recommend using Flamegraph tool to generate an interactive .svg file:
    # git clone https://github.com/brendangregg/FlameGraph
    # cd FlameGraph
    # ./flamegraph.pl --title "CUDA time" --countname "us." /tmp/profiler_stacks.txt > perf_viz.svg

if args.with_profiling:
    print("training with profiling")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        use_cuda=True,
        with_stack=True,
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=1,
            repeat=1),
        on_trace_ready=trace_handler) as profiler:
        with record_function("train"):
            train(profiler=profiler)
else:
    print("training without profiling")
    train()

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






