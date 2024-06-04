import tkinter as tk
import wave
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz, lfilter


class SignalProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processor")

        # UI elements
        self.load_button = tk.Button(root, text="Load WAV File", command=self.load_wav)
        self.load_button.pack()

        self.filter_type_label = tk.Label(root, text="Filter Type (low/high/band):")
        self.filter_type_label.pack()
        self.filter_type_entry = tk.Entry(root)
        self.filter_type_entry.pack()

        self.cutoff_label = tk.Label(root, text="Cutoff Frequency (for 'band' enter as 'low,high'):")
        self.cutoff_label.pack()
        self.cutoff_entry = tk.Entry(root)
        self.cutoff_entry.pack()

        self.process_button = tk.Button(root, text="Process", command=self.process_signal)
        self.process_button.pack()

        self.fig, self.axs = plt.subplots(2, 2)
        self.fig.tight_layout()
        plt.show(block=False)

        self.samplerate = None
        self.data = None

    def load_wav(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            with wave.open(file_path, 'rb') as wf:
                self.samplerate = wf.getframerate()
                nframes = wf.getnframes()
                data = wf.readframes(nframes)
                self.data = np.frombuffer(data, dtype=np.int16)

    def butter_filter(self, data, cutoff, fs, btype, order=5):
        nyquist = 0.5 * fs
        if btype == 'band':
            low, high = map(float, cutoff.split(','))
            normal_cutoff = [low / nyquist, high / nyquist]
        else:
            normal_cutoff = float(cutoff) / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        y = lfilter(b, a, data)
        return y

    def process_signal(self):
        if self.data is None or self.samplerate is None:
            print("No data loaded.")
            return

        filter_type = self.filter_type_entry.get()
        cutoff = self.cutoff_entry.get()

        if filter_type not in ['low', 'high', 'band']:
            print("Invalid filter type.")
            return

        filtered_data = self.butter_filter(self.data, cutoff, self.samplerate, btype=filter_type)

        self.axs[0, 0].clear()
        self.axs[0, 0].plot(self.data)
        self.axs[0, 0].set_title("Input Signal")

        self.axs[0, 1].clear()
        self.axs[0, 1].plot(filtered_data)
        self.axs[0, 1].set_title("Output Signal")

        self.axs[1, 0].clear()
        freqs, h = freqz(self.data)
        self.axs[1, 0].plot(freqs, np.abs(h))
        self.axs[1, 0].set_title("Input Frequency Response")

        self.axs[1, 1].clear()
        freqs, h = freqz(filtered_data)
        self.axs[1, 1].plot(freqs, np.abs(h))
        self.axs[1, 1].set_title("Output Frequency Response")

        plt.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()
