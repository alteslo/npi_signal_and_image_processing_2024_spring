import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import wave

def load_wav_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    with wave.open(file_path, 'rb') as wav_file:
        # Read the WAV file data
        # Processing code goes here


def apply_dsp(input_signal, choice):
    if choice == 1:
        output_signal = signal.lfilter([1, 0, -0.5], [1], input_signal)
    elif choice == 2:
        output_signal = signal.lfilter([1, 0, -0.5, 0.5], [1], input_signal)
    else:
        output_signal = signal.lfilter([1, 0.5, 0.5, 0], [1], input_signal)
    return output_signal

def plot_graphs(input_signal, output_signal):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(input_signal, label='Input Signal')
    plt.plot(output_signal, label='Output Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    freq, response = signal.freqz(input_signal)
    plt.plot(freq, abs(response), label='Input Signal')
    freq, response = signal.freqz(output_signal)
    plt.plot(freq, abs(response), label='Output Signal')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.show()

# GUI setup
root = tk.Tk()
root.title("Digital Signal Processing Simulation")

input_signal = None

def choose_dsp():
    global input_signal
    # Choose the type of digital signal processing procedure
    choice = 1  # Choose the procedure number here
    output_signal = apply_dsp(input_signal, choice)
    plot_graphs(input_signal, output_signal)

load_button = tk.Button(root, text="Load WAV File", command=load_wav_file)
load_button.pack()

dsp_button = tk.Button(root, text="Apply DSP", command=choose_dsp)
dsp_button.pack()

root.mainloop()