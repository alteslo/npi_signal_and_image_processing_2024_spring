import tkinter as tk
from tkinter import filedialog

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


class SignalSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Segmentation App")

        self.load_button = tk.Button(root, text="Load WAV File", command=self.load_wav_file)
        self.load_button.pack()

        self.model_button = tk.Button(root, text="Model Signal", command=self.model_signal)
        self.model_button.pack()

        self.segment_button = tk.Button(root, text="Segment Signal", command=self.segment_signal)
        self.segment_button.pack()

    def load_wav_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.signal, self.sr = librosa.load(file_path, sr=None)
            self.plot_signal(self.signal, title="Loaded Signal")

    def model_signal(self):
        duration = 5  # seconds
        self.sr = 22050  # sample rate
        t = np.linspace(0, duration, int(self.sr*duration), endpoint=False)
        self.signal = 15 * np.cos(2 * 2 * np.pi * t) + 3 * np.cos(20 * np.pi * t)
        self.plot_signal(self.signal, title="Modeled Signal")

    def segment_signal(self):
        energy = np.abs(self.signal) ** 2
        threshold = np.mean(energy) * 1.5
        segments = energy > threshold
        segment_boundaries = np.diff(segments.astype(int))
        segment_start_indices = np.where(segment_boundaries == 1)[0]
        segment_end_indices = np.where(segment_boundaries == -1)[0]

        self.plot_signal(self.signal, title="Segmented Signal", segment_boundaries=(segment_start_indices, segment_end_indices))

        self.reference_vectors = []
        for start, end in zip(segment_start_indices, segment_end_indices):
            segment = self.signal[start:end]
            if len(segment) < 1024:
                continue
            avg_freq, peak_freq = self.calculate_spectral_features(segment)
            if avg_freq is not None and peak_freq is not None:
                self.reference_vectors.append((avg_freq, peak_freq))
                print(f'Segment {start}-{end}: Avg Freq = {avg_freq}, Peak Freq = {peak_freq}')

    def calculate_spectral_features(self, segment):
        f, Pxx = scipy.signal.welch(segment, fs=self.sr, nperseg=1024)
        if len(Pxx) == 0:
            return None, None
        avg_freq = np.sum(f * Pxx) / np.sum(Pxx)
        peak_freq = f[np.argmax(Pxx)]
        return avg_freq, peak_freq

    def plot_signal(self, signal, title="Signal", segment_boundaries=None):
        plt.figure()
        plt.plot(np.arange(len(signal)) / self.sr, signal)
        plt.title(title)
        if segment_boundaries:
            segment_start_indices, segment_end_indices = segment_boundaries
            for start, end in zip(segment_start_indices, segment_end_indices):
                plt.axvline(start / self.sr, color='r')
                plt.axvline(end / self.sr, color='g')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalSegmentationApp(root)
    root.mainloop()
