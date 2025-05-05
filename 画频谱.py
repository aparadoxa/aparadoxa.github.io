#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_spectrogram.py

Usage:
    python generate_spectrogram.py [--input /path/to/audio.wav]

If no input path is provided, a file dialog will open for you to pick the audio file.
This script loads an audio file, computes its Mel spectrogram, converts it to dB scale,
and directly displays the spectrogram in a window (e.g., within PyCharm).
"""

import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and display a Mel spectrogram from an audio file."
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to input audio file (e.g., WAV, MP3). If omitted, a file dialog will appear.",
    )
    return parser.parse_args()


def select_file_dialog():
    if tk is None:
        raise RuntimeError("tkinter is required for file dialog but is not available.")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select audio file",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*")]
    )
    root.destroy()
    return file_path


def find_audio_file(directory, extensions=None):
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in extensions:
                return os.path.join(root, f)
    return None


def generate_and_show_spectrogram(path):
    # Load audio
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"Loaded '{path}' (duration: {duration:.2f}s, sr={sr}Hz)")

    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Display
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=512,
        x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (dB)')
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    input_path = args.input
    if not input_path:
        print("No input file specified, opening file dialog...")
        input_path = select_file_dialog()
        if not input_path:
            print("No file selected. Exiting.")
            return

    if not os.path.isfile(input_path):
        print(f"Specified input path does not exist: {input_path}")
        print("Scanning current directory for audio files...")
        found = find_audio_file(os.getcwd())
        if found:
            print(f"Found audio file: {found}")
            input_path = found
        else:
            print("No audio file found. Exiting.")
            return

    generate_and_show_spectrogram(input_path)


if __name__ == "__main__":
    main()
