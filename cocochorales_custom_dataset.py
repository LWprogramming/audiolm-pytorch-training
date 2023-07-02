from torch.utils.data import Dataset
from pathlib import Path
from torchaudio.functional import resample


class CocochoralesCustomDataset(Dataset):
	"""Copied from Cocochorales documentation here: https://github.com/lukewys/chamber-ensemble-generator/blob/a8db78a49661ea9d72341bef61fff9376696afa4/data_format.md

	Each track has its own unique ID. The naming of the tracks follows <ensemble>_track<id>. Where <ensemble> is one of four ensembles: string, brass, woodwind, random. The ID is a 6-digit number, from 000001 to 240000, where 000001-192000 are training set, 192001-216000 are valid set, and 216001-240000 are test set.

	We take the stems_audio folder, which looks like

 	stems_audio/
	|-- 0_violin.wav
    |-- 1_violin.wav
    |-- 2_viola.wav
    |-- 3_cello.wav

    The `__getitem__` method processes audio files prefixed with `0_` and `3_`. It trims and aligns the audio data to create a sequence of equal length from both files, separated by a configurable (but default half-second) of silence. This allows transformers to learn harmonies from two parallel parts.
	"""
	def __init__(self, folder, max_length, target_sample_hz, seq_len_multiple_of, silence_length_seconds=0.5):
		super().__init__()
		path = Path(folder)
		assert path.exists(), 'folder does not exist'

		# files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
		stem_audio_folders = [file for file in path.glob(f'**/stems_audio')]
		assert len(stem_audio_folders) > 0, 'no sound files found'

		self.stem_audio_folders = stem_audio_folders

		self.target_sample_hz = cast_tuple(target_sample_hz)
		num_outputs = len(self.target_sample_hz)

		self.max_length = cast_tuple(max_length, num_outputs)
		min_seconds = 30
		for max_length in self.max_length:
			# ensure max_length is long enough so we can learn accompaniment over a long time range
			assert max_length >= min_seconds * self.target_sample_hz, f"max_length must be at least {min_seconds} seconds * target_sample_hz"
		self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)
		assert len(self.max_length) == len(self.target_sample_hz) == len(self.seq_len_multiple_of)

		# now calculate the length of the audio from a single source
		self.silence_length_samples = tuple(int(silence_length_seconds * target_sample_hz) for target_sample_hz in self.target_sample_hz)
		# assumes everything divides evenly
		self.num_samples_from_each_track = tuple((max_length - silence_length_samples) // 2 for max_length, silence_length_samples in zip(self.max_length, self.silence_length_samples))

	def __len__(self):
		return len(self.stem_audio_folders)

	def get_audio_data(self, file):
		# given pathlib glob, return full resampled data and original sample rate
		data, sample_hz = torchaudio.load(file)
		assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'
		if data.shape[0] > 1:
			# the audio has more than 1 channel, convert to mono
			data = torch.mean(data_melody, dim=0, keepdim=True)

		# resample if target_sample_hz is not None in the tuple
		data_tuple = tuple(
			(resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in
			zip(data, self.target_sample_hz))
		return data_tuple, sample_hz

	def __getitem__(self, idx):
		# coding todo
		# then getitem will just take the stems_audio folder, load the 0_ and 3_ wav files (note they might not always match violin cello etc, but they should always have 0_ and 3_), and when trimming the audio we actually process it a little differently than in the given class. here's how:
		# suppose 0_ is represented by samples a0 a1 a2 a3 ... etc and 3_ is represented by b0 b1 b2 b3 ... etc
		# then we want to have the trimmed data to be something like (suppose the total length for each item is n, so the resulting trimmed data (and aligned to seq_len_multiple_of) has exactly n samples)-- and suppose half a second of samples is m samples:
		# then we want the resulting sample to be a0 a1 a2 ... a((n - m)/2) 0 0 0 ... (m zeroes total) 0 b0 b1 b2 ... b((n - m) /2)
		# In other words, the result is exactly n samples long, and within that result we want an equally long sub-sequence from a and b, separated by about a half second of zeroes which should be silence in wav. If they don't divide perfectly, that's ok, make sure the code accounts for that-- just make the silence slightly longer as needed. the crucial part is that the part from 0_* files and 3_* files are the exact same length and separated by zeroes. everything else is the same.
		# this lets the transformers learn on two parallel parts and hopefully harmonies
		# example of code:
		# data, sample_hz = torchaudio.load(file)
		# data = torch.cat((data[..., :200000], torch.zeros(1, 30000), data[..., 200000:]), 1)

		folder = self.stem_audio_folders[idx]
		melody_file = folder.glob(f'0_*.wav')
		harmony_file = folder.glob(f'3_*.wav')
		data_melody_tuple, sample_hz_melody = self.get_audio_data(melody_file)
		data_harmony_tuple, sample_hz_harmony = self.get_audio_data(harmony_file)

		# probably 16kHz
		assert sample_hz_melody == sample_hz_harmony, 'sample_hz_melody and sample_hz_harmony must be the same'
		for resampled_melody, resampled_harmony in zip(data_melody_tuple, data_harmony_tuple):
			assert resampled_melody.shape == resampled_harmony.shape, f'resampled_melody and resampled_harmony must have the same shape but found resampled_melody.shape={resampled_melody.shape} and resampled_harmony.shape={resampled_harmony.shape}'

		num_outputs = len(self.target_sample_hz)

		output = []

		# process each of the data resample at different frequencies individually

		for data_melody, data_harmony, num_samples_from_each_track, seq_len_multiple_of in zip(data_melody_tuple, data_harmony_tuple, self.num_samples_from_each_track, self.seq_len_multiple_of):
			audio_length = data_melody.size(1)

			# pad or curtail

			if audio_length > num_samples_from_each_track:
				max_start = audio_length - num_samples_from_each_track
				start = torch.randint(0, max_start, (1,))
				data_melody = data_melody[:, start:start + num_samples_from_each_track]
				data_harmony = data_harmony[:, start:start + num_samples_from_each_track]

			else:
				data_melody = F.pad(data_melody, (0, num_samples_from_each_track - audio_length), 'constant')
				data_harmony = F.pad(data_harmony, (0, num_samples_from_each_track - audio_length), 'constant')

			data_melody = rearrange(data_melody, '1 ... -> ...')
			data_harmony = rearrange(data_harmony, '1 ... -> ...')

			if exists(max_length):
				data = data[:max_length]

			if exists(seq_len_multiple_of):
				data = curtail_to_multiple(data, seq_len_multiple_of)

			output.append(data.float())

		# cast from list to tuple

		output = tuple(output)

		# return only one audio, if only one target resample freq

		if num_outputs == 1:
			return output[0]

		return output




"""
class SoundDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav', 'mp3', 'webm'],
        max_length: OptionalIntOrTupleInt = None,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        self.max_length = cast_tuple(max_length, num_outputs)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        # resample if target_sample_hz is not None in the tuple

        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, self.target_sample_hz))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            # pad or curtail

            if audio_length > max_length:
                max_start = audio_length - max_length
                start = torch.randint(0, max_start, (1, ))
                data = data[:, start:start + max_length]

            else:
                data = F.pad(data, (0, max_length - audio_length), 'constant')

            data = rearrange(data, '1 ... -> ...')

            if exists(max_length):
                data = data[:max_length]

            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output
"""