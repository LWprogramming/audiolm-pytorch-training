# imports
import math
import wave
import struct
import os
import urllib.request
# import tarfile
from audiolm_pytorch import AudioLMSoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from torch import nn
import torch
import torchaudio
from torch.profiler import profile, record_function, ProfilerActivity
import datetime
# import boto3
# import datetime
# from botocore.errorfactory import ClientError

# define all dataset paths, checkpoints, etc
prefix = "/fsx/itsleonwu/audiolm-pytorch-results"
dataset_folder = f"{prefix}/placeholder_dataset"
hubert_ckpt = f'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer

# Placeholder data generation
def get_sinewave(freq=440.0, duration_ms=200, volume=1.0, sample_rate=16000.0):
  # code adapted from https://stackoverflow.com/a/33913403
  audio = []
  num_samples = duration_ms * (sample_rate / 1000.0)
  for x in range(int(num_samples)):
    audio.append(volume * math.sin(2 * math.pi * freq * (x / sample_rate)))
  return audio

def save_wav(file_name, audio, sample_rate=16000.0):
  # Open up a wav file
  wav_file=wave.open(file_name,"w")
  # wav params
  nchannels = 1
  sampwidth = 2
  # 16000 is the industry standard sample rate - CD quality.  If you need to
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
  save_wav(f"{dataset_folder}/example.wav", get_sinewave())
  save_wav(f"{dataset_folder}/example2.wav", get_sinewave(duration_ms=500))
  os.makedirs(f"{dataset_folder}/subdirectory")
  save_wav(f"{dataset_folder}/subdirectory/example.wav", get_sinewave(freq=330.0))
make_placeholder_dataset()

#######

# soundstream = AudioLMSoundStream(
#     codebook_size = 1024,
#     rq_num_quantizers = 8,
#     attn_window_size = 128,       # local attention receptive field at bottleneck
#     attn_depth = 2                # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
# )

# soundstream_trainer = SoundStreamTrainer(
#     soundstream,
#     folder = dataset_folder,
#     lr=3e-4,
#     batch_size = 4,
#     grad_accum_every = 8, # effective batch size of batch_size * grad_accum_every = 32
#     data_max_length_seconds = 2,  # train on 2 second audio
#     results_folder = f"{prefix}/soundstream_results",
#     save_results_every = 4,
#     save_model_every = 4,
#     num_train_steps = 9
# ).cuda()

# soundstream_trainer.train() # skip soundstream for now

#############

wav2vec = HubertWithKmeans(
    checkpoint_path = f"{prefix}/{hubert_ckpt}",
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
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1,
    results_folder = f"{prefix}/semantic_results",
)

semantic_trainer.train()

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
    soundstream = soundstream,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    results_folder = f"{prefix}/coarse_results",
    save_results_every = 4,
    save_model_every = 4,
    num_train_steps = 9
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

coarse_trainer.train()

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
    soundstream = soundstream,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    save_results_every = 4,
    save_model_every = 4,
    num_train_steps = 9,
    results_folder = f"{prefix}/fine_results",
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

fine_trainer.train()

################
# Everything together

audiolm = AudioLM(
    wav2vec = wav2vec,
    soundstream = soundstream,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer
)

generated_wav = audiolm(batch_size = 1)
output_path = f"{prefix}/out.wav"
sample_rate = 16000
torchaudio.save(output_path, generated_wav.cpu(), sample_rate)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True) as prof:
#     with record_function("model_inference"):
#         generated_wav = audiolm(batch_size = 1)
#         output_path = f"{prefix}/out.wav"
#         sample_rate = 16000
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






