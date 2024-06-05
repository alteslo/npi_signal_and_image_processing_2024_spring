import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import freqz


class CFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Процедуры ЦФ")

        self.filepath = None

        # UI Elements
        self.load_button = ttk.Button(root, text="Загрузить WAV файл", command=self.load_wav_file)
        self.load_button.pack(pady=10)

        self.filter_label = ttk.Label(root, text="Выберите тип ЦФ:")
        self.filter_label.pack()

        self.filter_var = tk.StringVar(value="1")
        self.filter_menu = ttk.Combobox(root, textvariable=self.filter_var)
        self.filter_menu['values'] = [str(i) for i in range(1, 11)]
        self.filter_menu.pack(pady=10)

        self.plot_button = ttk.Button(root, text="Показать графики", command=self.plot_signals)
        self.plot_button.pack(pady=10)

    def load_wav_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    def plot_signals(self):
        if not self.filepath:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите WAV файл.")
            return

        filter_type = int(self.filter_var.get())
        fs, data = wavfile.read(self.filepath)

        if data.ndim > 1:
            data = data[:, 0]  # Используем только один канал для простоты

        # Нормализация данных
        data = data / np.max(np.abs(data), axis=0)

        # Применение ЦФ
        output_signal = self.apply_filter(data, filter_type)

        # Построение графиков
        self.plot_waveforms(data, output_signal, fs)
        self.plot_frequency_response(data, output_signal, fs)

    def apply_filter(self, signal, filter_type):
        output_signal = np.zeros_like(signal)
        if filter_type == 1:
            for n in range(1, len(signal)):
                output_signal[n] = signal[n] - 0.5 * signal[n - 1]
        elif filter_type == 2:
            for n in range(2, len(signal)):
                output_signal[n] = signal[n] - 0.5 * signal[n - 1] + 0.5 * signal[n - 2]
        elif filter_type == 3:
            for n in range(2, len(signal)):
                output_signal[n] = signal[n] + 0.5 * signal[n - 1] + 0.5 * signal[n - 2]
        elif filter_type == 4:
            for n in range(2, len(signal)):
                output_signal[n] = 0.5 * signal[n] + 0.25 * signal[n - 1] + 0.25 * signal[n - 2]
        elif filter_type == 5:
            for n in range(1, len(signal)):
                output_signal[n] = signal[n] - 0.5 * signal[n - 1] + 0.5 * output_signal[n - 1]
        elif filter_type == 6:
            for n in range(2, len(signal)):
                output_signal[n] = signal[n] + 0.5 * signal[n - 1] - signal[n - 2] - 0.5 * output_signal[n - 1]
        elif filter_type == 7:
            for n in range(2, len(signal)):
                output_signal[n] = signal[n] + 0.5 * signal[n - 1] - 0.5 * signal[n - 2] - 0.5 * output_signal[n - 1]
        elif filter_type == 8:
            for n in range(2, len(signal)):
                output_signal[n] = signal[n] - 0.5 * signal[n - 1] - 0.5 * signal[n - 2] - 0.5 * output_signal[n - 1]
        elif filter_type == 9:
            for n in range(2, len(signal)):
                output_signal[n] = signal[n] + 0.5 * signal[n - 1] + 0.5 * signal[n - 2] - 0.5 * output_signal[n - 1]
        elif filter_type == 10:
            for n in range(2, len(signal)):
                output_signal[n] = signal[n] - 0.5 * signal[n - 1] + 0.5 * signal[n - 2] + 0.5 * output_signal[n - 1]
        return output_signal

    def plot_waveforms(self, input_signal, output_signal, fs):
        t = np.arange(len(input_signal)) / fs
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(input_signal)
        # plt.plot(t, input_signal)
        plt.title("Входной сигнал")
        plt.subplot(2, 1, 2)
        plt.plot(output_signal)
        plt.plot(t, output_signal)
        plt.title("Выходной сигнал")
        plt.tight_layout()
        plt.show()

    def plot_frequency_response(self, input_signal, output_signal, fs):
        f_input, h_input = freqz(input_signal, fs=fs)
        f_output, h_output = freqz(output_signal, fs=fs)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(f_input, 20 * np.log10(np.abs(h_input)))
        plt.title("АЧХ Входного сигнала")
        plt.subplot(2, 1, 2)
        plt.plot(f_output, 20 * np.log10(np.abs(h_output)))
        plt.title("АЧХ Выходного сигнала")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CFApp(root)
    root.mainloop()
