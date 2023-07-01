import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from pathlib import Path
import argparse


def plot_file(file):
	sample_rate, samples = wavfile.read(file)
	frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
	plt.pcolormesh(times, frequencies, np.log(spectrogram))
	# plt.imshow(spectrogram)
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

def plot_dir(dir, exts=("wav",)):
	path = Path(dir)
	print(dir)
	files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
	print(len(files))
	for file in files:
		print(file)
		plot_file(file)

parser = argparse.ArgumentParser()
parser.add_argument('--abs_dir_path', type=str)
args = parser.parse_args()
plot_dir(args.abs_dir_path)
