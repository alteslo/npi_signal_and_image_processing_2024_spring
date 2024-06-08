import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import freqz


class CFApp:
    """
    Основной класс приложения,
    который отвечает за создание интерфейса и выполнение основных функций.
    """
    def __init__(self, root):
        """
        Инициализирует основной интерфейс приложения, создает кнопки и меню.

        :param root: главный элемент окна Tkinter
        """
        self.root = root
        self.root.title("Процедуры ЦФ")

        self.filepath = None

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
        """
        Открывает диалоговое окно для выбора WAV файла и сохраняет путь к нему в переменной self.filepath
        """
        self.filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    def plot_signals(self):
        """
        Проверяет, загружен ли файл. Если файл не загружен, выводит сообщение об ошибке.
        Читает WAV файл, нормализует данные и применяет выбранный фильтр.
        Вызывает функции для построения графиков временных и частотных характеристик
        """
        if not self.filepath:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите WAV файл.")
            return

        filter_type = int(self.filter_var.get())
        fs, data = wavfile.read(self.filepath)

        if data.ndim > 1:
            data = data[:, 0]  # Используем только один канал для простоты

        data = data / np.max(np.abs(data), axis=0)  # Нормализация данных
        output_signal = self.apply_filter(data, filter_type)  # Применение ЦФ

        # Построение графиков
        self.plot_waveforms(data, output_signal, fs)
        self.plot_frequency_response(data, output_signal, fs)

    def apply_filter(self, signal, filter_type):
        """
        Применяет выбранный цифровой фильтр к сигналу

        :param signal: входной аудиосигнал
        :param filter_type: тип фильтра, выбранный пользователем
        :return: выходной сигнал после фильтрации
        """
        output_signal = np.zeros_like(signal)
        match filter_type:
            case 1:
                for n in range(1, len(signal)):
                    output_signal[n] = signal[n] - 0.5 * signal[n - 1]
            case 2:
                for n in range(2, len(signal)):
                    output_signal[n] = signal[n] - 0.5 * signal[n - 1] + 0.5 * signal[n - 2]
            case 3:
                for n in range(2, len(signal)):
                    output_signal[n] = signal[n] + 0.5 * signal[n - 1] + 0.5 * signal[n - 2]
            case 4:
                for n in range(2, len(signal)):
                    output_signal[n] = 0.5 * signal[n] + 0.25 * signal[n - 1] + 0.25 * signal[n - 2]
            case 5:
                for n in range(1, len(signal)):
                    output_signal[n] = signal[n] - 0.5 * signal[n - 1] + 0.5 * output_signal[n - 1]
            case 6:
                for n in range(2, len(signal)):
                    output_signal[n] = signal[n] + 0.5 * signal[n - 1] - signal[n - 2] - 0.5 * output_signal[n - 1]
            case 7:
                for n in range(2, len(signal)):
                    output_signal[n] = signal[n] + 0.5 * signal[n - 1] - 0.5 * signal[n - 2] - 0.5 * output_signal[n - 1]
            case 8:
                for n in range(2, len(signal)):
                    output_signal[n] = signal[n] - 0.5 * signal[n - 1] - 0.5 * signal[n - 2] - 0.5 * output_signal[n - 1]
            case 9:
                for n in range(2, len(signal)):
                    output_signal[n] = signal[n] + 0.5 * signal[n - 1] + 0.5 * signal[n - 2] - 0.5 * output_signal[n - 1]
            case 10:
                for n in range(2, len(signal)):
                    output_signal[n] = signal[n] - 0.5 * signal[n - 1] + 0.5 * signal[n - 2] + 0.5 * output_signal[n - 1]
        return output_signal

    def plot_waveforms(self, input_signal, output_signal, fs):
        """
        Строит графики временных характеристик входного и выходного сигналов. 

        :param input_signal: входной сигнал
        :param output_signal: выходной сигнал после фильтрации
        :param fs: частота дискретизации
        """
        t = np.arange(len(input_signal)) / fs
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(input_signal)
        plt.plot(t, input_signal)
        plt.title("Входной сигнал")
        plt.subplot(2, 1, 2)
        plt.plot(output_signal)
        plt.plot(t, output_signal)
        plt.title("Выходной сигнал")
        plt.tight_layout()
        plt.show()

    def plot_frequency_response(self, input_signal, output_signal, fs):
        """
        Строит графики частотных характеристик входного и выходного сигналов.

        :param input_signal: входной сигнал
        :param output_signal: выходной сигнал после фильтрации
        :param fs: частота дискретизации
        """
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
