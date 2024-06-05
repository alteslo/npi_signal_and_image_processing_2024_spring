import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks, correlate
from scipy.fftpack import fft


class SignalSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сегментация сигнала")

        self.filepath = None
        self.sampling_rate = None
        self.signal = None

        # UI Elements
        self.load_button = ttk.Button(root, text="Загрузить WAV файл", command=self.load_wav_file)
        self.load_button.pack(pady=10)

        self.model_button = ttk.Button(root, text="Моделировать сигнал", command=self.model_signal)
        self.model_button.pack(pady=10)

        self.segment_button = ttk.Button(root, text="Сегментировать сигнал", command=self.segment_signal)
        self.segment_button.pack(pady=10)

        self.plot_button = ttk.Button(root, text="Показать графики", command=self.plot_signals)
        self.plot_button.pack(pady=10)

        self.segments = None
        self.segment_boundaries = []

    def load_wav_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.filepath:
            self.sampling_rate, data = wavfile.read(self.filepath)
            if data.ndim > 1:
                # Усредняем каналы, если сигнал многоканальный
                self.signal = np.mean(data, axis=1)
            else:
                self.signal = data
            self.signal = self.signal / np.max(np.abs(self.signal), axis=0)
            print(f'Загружен файл: {self.filepath}')
            print(f'Частота дискретизации: {self.sampling_rate} Гц')
            print(f'Длина сигнала: {len(self.signal)} отсчетов')

    def model_signal(self):
        duration = 5  # seconds
        self.sampling_rate = 44100  # Hz
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        self.signal = 15 * np.cos(2 * 2 * np.pi * t) + 3 * np.cos(20 * np.pi * t)
        self.signal = self.signal / np.max(np.abs(self.signal), axis=0)
        print(f'Смоделирован сигнал длительностью {duration} секунд')

    def segment_signal(self):
        # Простой алгоритм сегментации по шаблону
        duration = 1  # seconds
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        template = 15 * np.cos(2 * 2 * np.pi * t) + 3 * np.cos(20 * np.pi * t)
        template = template / np.max(np.abs(template), axis=0)

        correlation = correlate(self.signal, template, mode='valid')
        peaks, _ = find_peaks(correlation, height=np.max(correlation) * 0.5)

        self.segment_boundaries = [0] + peaks.tolist() + [len(self.signal)]
        self.segments = [self.signal[self.segment_boundaries[i]:self.segment_boundaries[i + 1]] for i in range(len(self.segment_boundaries) - 1)]
        print(f'Найдено {len(self.segments)} сегментов')

    def plot_signals(self):
        if self.signal is None:
            tk.messagebox.showerror("Ошибка", "Пожалуйста, загрузите или смоделируйте сигнал.")
            return

        # Построение исходного сигнала
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.signal)
        plt.title("Исходный сигнал")

        # Построение сегментированного сигнала
        plt.subplot(2, 1, 2)
        for boundary in self.segment_boundaries:
            plt.axvline(x=boundary, color='r')
        plt.plot(self.signal)
        plt.title("Сегментированный сигнал с метками границ")

        plt.tight_layout()
        plt.show()

    def compute_reference_vectors(self):
        if not self.segments:
            print("Сегменты не найдены.")
            return

        ref_vectors = []
        for segment in self.segments:
            # Пример вычисления эталонного вектора: математическое ожидание и дисперсия
            mean = np.mean(segment)
            variance = np.var(segment)
            ref_vectors.append((mean, variance))
            print(f'Эталонный вектор: {mean}, {variance}')

        return ref_vectors


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalSegmentationApp(root)
    root.mainloop()
