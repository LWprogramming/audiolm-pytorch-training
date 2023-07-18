from pathlib import Path
from functools import partial, wraps

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

import torchaudio
from torchaudio.functional import resample

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# from audiolm_pytorch.utils import curtail_to_multiple

from einops import rearrange

# this code is heavily heavily modeled after https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/data.py

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

class CocochoralesCustomDataset(Dataset):
    """Copied from Cocochorales documentation here: https://github.com/lukewys/chamber-ensemble-generator/blob/a8db78a49661ea9d72341bef61fff9376696afa4/data_format.md

    Each track has its own unique ID. The naming of the tracks follows <ensemble>_track<id>. Where <ensemble> is one of four ensembles: string, brass, woodwind, random. The ID is a 6-digit number, from 000001 to 240000, where 000001-192000 are training set, 192001-216000 are valid set, and 216001-240000 are test set.


    This code is closely coupled to the cocochorales_downloader.sh script, which modifies the format of the data in each track. Each track folder (after the downloader script is done) contains 4 wav files.

    For example, the string_track000001 folder contains the following files:
    |-- 1_violin.wav
    |-- 2_violin.wav
    |-- 3_viola.wav
    |-- 4_cello.wav
    These wav files used to be in the stem_audio subfolder, but the downloader script moves them up one level and cleans out the other files and folders.

    The `__getitem__` method processes audio files prefixed with `0_` and `3_`. It trims and aligns the audio data to create a sequence of equal length from both files, separated by a configurable (but default half-second) of silence. This allows transformers to learn harmonies from two parallel parts.

    We also trim off the first and last half-second of each file (the audio around the edges of the sound file in the original dataset seem a little sketchy)
    """
    def __init__(self, folder, max_length, target_sample_hz, silence_length_seconds=0.5):
        # intentionally leaving out seq_len_multiple_of which exists in the original AudioLM repo because I don't want to have to deal with it
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        # files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        stem_audio_folders = [file for file in path.glob(f'**/*track*')]
        assert len(stem_audio_folders) > 0, 'no sound files found'

        self.stem_audio_folders = stem_audio_folders

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        assert max_length is not None, "max_length must be specified"
        self.max_length = cast_tuple(max_length, num_outputs)
        min_seconds = 30
        for i in range(num_outputs):
            # ensure max_length is long enough so we can learn accompaniment over a long time range
            assert self.max_length[i] >= min_seconds * self.target_sample_hz[i], f"max_length must be at least {min_seconds} seconds * target_sample_hz"
        # self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)
        # assert len(self.max_length) == len(self.target_sample_hz) == len(self.seq_len_multiple_of)

        # now calculate the length of the audio from a single source
        self.silence_num_samples = tuple(int(silence_length_seconds * target_sample_hz) for target_sample_hz in self.target_sample_hz)
        # assumes everything divides evenly
        self.num_samples_from_each_track = tuple((max_length - silence_length_samples) // 2 for max_length, silence_length_samples in zip(self.max_length, self.silence_num_samples))

    def __len__(self):
        return len(self.stem_audio_folders)

    def get_audio_data(self, file):
        # given pathlib glob, return full resampled data and original sample rate after trimming off the first and last half-second as described in the docstring
        data, sample_hz = torchaudio.load(file)
        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'
        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0) # 1 x samples shape
        num_outputs = len(self.target_sample_hz)

        # trim off the first and last half-second:
        num_samples_to_trim = int(sample_hz * 0.5)
        data = data[:, num_samples_to_trim:-num_samples_to_trim]

        # prepare data into tuple so we can resample per target_sample_hz entry
        data = cast_tuple(data, num_outputs)
        # print(f"data shape is {data[0].shape}")
        # resample if target_sample_hz is not None in the tuple
        data_tuple = tuple(
            (resample(d, sample_hz, target_sample_hz) if target_sample_hz is not None else d) for d, target_sample_hz in
            zip(data, self.target_sample_hz))
        # print(f"datatuple first shape is {data_tuple[0].shape}")
        return data_tuple, sample_hz

    def __getitem__(self, idx):
        folder = self.stem_audio_folders[idx]
        melody_file = next(folder.glob(f'1_*.wav')) # should only be one file
        harmony_file = next(folder.glob(f'4_*.wav'))
        data_melody_tuple, sample_hz_melody = self.get_audio_data(melody_file)
        data_harmony_tuple, sample_hz_harmony = self.get_audio_data(harmony_file)

        # probably 16kHz
        assert sample_hz_melody == sample_hz_harmony, 'sample_hz_melody and sample_hz_harmony must be the same'
        for resampled_melody, resampled_harmony in zip(data_melody_tuple, data_harmony_tuple):
            assert resampled_melody.shape == resampled_harmony.shape, f'resampled_melody and resampled_harmony must have the same shape but found resampled_melody.shape={resampled_melody.shape} and resampled_harmony.shape={resampled_harmony.shape}'

        num_outputs = len(self.target_sample_hz)

        output = []

        # process each of the data resample at different frequencies individually

        for data_melody_at_curr_hz, data_harmony_at_curr_hz, num_samples_at_curr_hz, silence_num_samples_at_curr_hz in zip(data_melody_tuple, data_harmony_tuple, self.num_samples_from_each_track, self.silence_num_samples):
            audio_length = data_melody_at_curr_hz.size(1)

            # pad or curtail

            if audio_length > num_samples_at_curr_hz:
                max_start = audio_length - num_samples_at_curr_hz
                start = torch.randint(0, max_start, (1,))
                # print("data_melody_at_curr_hz shape is ", data_melody_at_curr_hz.shape)
                data_melody_at_curr_hz = data_melody_at_curr_hz[:, start:start + num_samples_at_curr_hz]
                # print("if data_melody_at_curr_hz shape is ", data_melody_at_curr_hz.shape)
                data_harmony_at_curr_hz = data_harmony_at_curr_hz[:, start:start + num_samples_at_curr_hz]
            else:
                # print("data_melody_at_curr_hz shape is ", data_melody_at_curr_hz.shape)
                data_melody_at_curr_hz = F.pad(data_melody_at_curr_hz, (0, num_samples_at_curr_hz - audio_length), 'constant')
                # print("else data_melody_at_curr_hz shape is ", data_melody_at_curr_hz.shape)
                data_harmony_at_curr_hz = F.pad(data_harmony_at_curr_hz, (0, num_samples_at_curr_hz - audio_length), 'constant')

            data_melody_at_curr_hz = rearrange(data_melody_at_curr_hz, '1 ... -> ...')
            data_harmony_at_curr_hz = rearrange(data_harmony_at_curr_hz, '1 ... -> ...')
            # print(f"data_melody.shape={data_melody_at_curr_hz.shape} and data_harmony.shape={data_harmony_at_curr_hz.shape} with silence_length_samples={silence_num_samples_at_curr_hz}")

            # print(f"data_melody_at_curr_hz.shape={data_melody_at_curr_hz.shape}")
            to_append = torch.cat((data_melody_at_curr_hz, torch.zeros(silence_num_samples_at_curr_hz), data_harmony_at_curr_hz), dim=0).float()
            # print(f"to_append.shape={to_append.shape}") # should be 1-dimensional, just the length of the audio in samples.
            output.append(to_append)
            # print(f"output[-1].shape={output[-1].shape}")
        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output
def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

# copied from audiolm as a reference. we're not using this
# class SoundDataset(Dataset):
#     def __init__(
#         self,
#         folder,
#         exts = ['flac', 'wav', 'mp3', 'webm'],
#         max_length = None,
#         target_sample_hz = None,
#         # seq_len_multiple_of = None
#     ):
#         super().__init__()
#         path = Path(folder)
#         assert path.exists(), 'folder does not exist'
#
#         files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
#         assert len(files) > 0, 'no sound files found'
#
#         self.files = files
#
#         self.target_sample_hz = cast_tuple(target_sample_hz)
#         num_outputs = len(self.target_sample_hz)
#
#         self.max_length = cast_tuple(max_length, num_outputs)
#         # self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)
#
#         # assert len(self.max_length) == len(self.target_sample_hz) # == len(self.seq_len_multiple_of)
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         file = self.files[idx]
#
#         data, sample_hz = torchaudio.load(file)
#
#         assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'
#
#         if data.shape[0] > 1:
#             # the audio has more than 1 channel, convert to mono
#             data = torch.mean(data, dim=0).unsqueeze(0)
#
#         num_outputs = len(self.target_sample_hz)
#         data = cast_tuple(data, num_outputs)
#
#         # resample if target_sample_hz is not None in the tuple
#         data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if target_sample_hz is not None else d) for d, target_sample_hz in zip(data, self.target_sample_hz))
#
#         output = []
#
#         # process each of the data resample at different frequencies individually
#         # print(f"len data tuple is {len(data_tuple)}")
#         for data, max_length in zip(data_tuple, self.max_length):
#             audio_length = data.size(1)
#
#             # pad or curtail
#             # print(f"data shape is {data.shape}")
#             if audio_length > max_length:
#                 max_start = audio_length - max_length
#                 start = torch.randint(0, max_start, (1, ))
#                 data = data[:, start:start + max_length]
#
#             else:
#                 data = F.pad(data, (0, max_length - audio_length), 'constant')
#
#             data = rearrange(data, '1 ... -> ...')
#
#             if max_length is not None:
#                 data = data[:max_length]
#
#             # if seq_len_multiple_of is not None:
#             #     data = curtail_to_multiple(data, seq_len_multiple_of)
#             # print(f"dat float shape {data.float().shape}")
#             output.append(data.float())
#
#         # cast from list to tuple
#
#         output = tuple(output)
#
#         # return only one audio, if only one target resample freq
#
#         if num_outputs == 1:
#             return output[0]
#
#         return output

if __name__ == "__main__":
    dataset = CocochoralesCustomDataset(folder='/fsx/itsleonwu/audiolm-pytorch-datasets/cocochorales_main_dataset_v1/1', target_sample_hz=16000, max_length=16000*30)
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0, shuffle=True)
    for batch in dataloader:
        print(f"len batch is {len(batch)} and first elemtn has shape {batch[0].shape}") # one element of shape 1 x num_samples
        break


