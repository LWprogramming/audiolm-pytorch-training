from torch.utils.data import Dataset

class CocochoralesCustomDataset(Dataset):
	"""Copied from Cocochorales documentation here: https://github.com/lukewys/chamber-ensemble-generator/blob/a8db78a49661ea9d72341bef61fff9376696afa4/data_format.md

	Each track has its own unique ID. The naming of the tracks follows <ensemble>_track<id>. Where <ensemble> is one of four ensembles: string, brass, woodwind, random. The ID is a 6-digit number, from 000001 to 240000, where 000001-192000 are training set, 192001-216000 are valid set, and 216001-240000 are test set.

	We take the stems_audio folder, which looks like

 	stems_audio/
	|-- 0_violin.wav
    |-- 1_violin.wav
    |-- 2_viola.wav
    |-- 3_cello.wav

    And gives samples that are the first 

	"""
	pass

# coding todo
# write the dataset class for cocochorales implementing the necessary functions and based on the docstring, following the example class below with identical params-- if there are any functions that are called here that are undefined, point those out first and refuse to generate any code.
# we want files to be a variable something like self.tracks instead, which contains a list of paths to the various stems_audio folders. there's a good chance they're sequential but you can just path.glob them in case we're testing random tracks instead-- the one thing you can rely on is the stems_audio/ folder format in the docstring.
# then getitem will just take the stems_audio folder, load the 0_ and 3_ wav files (note they might not always match violin cello etc, but they should always have 0_ and 3_), and when trimming the audio we actually process it a little differently than in the given class. here's how:
# suppose 0_ is represented by samples a0 a1 a2 a3 ... etc and 3_ is represented by b0 b1 b2 b3 ... etc
# then we want to have the trimmed data to be something like (suppose the total length for each item is n, so the resulting trimmed data (and aligned to seq_len_multiple_of) has exactly n samples)-- and suppose half a second of samples is m samples:
# then we want the resulting sample to be a0 a1 a2 ... a((n - m)/2) 0 0 0 ... (m zeroes total) 0 b0 b1 b2 ... b((n - m) /2)
# In other words, the result is exactly n samples long, and within that result we want an equally long sub-sequence from a and b, separated by about a half second of zeroes which should be silence in wav. If they don't divide perfectly, that's ok, make sure the code accounts for that-- just make the silence slightly longer as needed. the crucial part is that the part from 0_* files and 3_* files are the exact same length and separated by zeroes. everything else is the same.
# this lets the transformers learn on two parallel parts and hopefully harmonies

# btw, the init should raise an assertion error if any max_length in self.max_length (after everything's casted to tuple) is less than like 5 seconds (taking into account the corresponding target sample hz for each one, so it's not just checking if <5, it's 5 * sampling frequency) because you can't really learn long range accompaniment like that i don't think
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