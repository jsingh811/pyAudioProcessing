#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:27:32 2021

@author: jsingh
"""

################################################################################
# Imports
################################################################################
import json
import os
import warnings
import numpy as np
from scipy.io import wavfile
warnings.filterwarnings("ignore")
from pydub import AudioSegment


################################################################################
# Globals
################################################################################

ST_WIN = 0.05
ST_STEP = 0.05

################################################################################
# Functions
################################################################################


def convert_audio_to_mono(signal):
    """
    Some audios contain a 2 dim array like
    [[0.34, 0.32], [0.01, 0.02], ...]
    corresponding to the audio split between left and right.
    For stansardized processing and the working of certain functionalities and
    computations, we'll convert it to a mono signal with single dimension by
    averaging the L and R values.
    """
    len_arr = len(np.array(signal[0], ndmin=1))

    if len_arr > 1:
        signal = sum([signal[:, i] for i in range(len_arr)]) / len_arr

    return signal


def write_to_json(file_name, data):
    """
    Write data to file_name.
    """
    with open(file_name, "w") as outfile:
        json.dump(data, outfile, indent=1)
    print("\nResults saved in {}\n".format(file_name))


def read_audio(input_file):
    """
    Reads input audio file and returns sampling frequency along with the signal.
    """
    sampling_rate = 0
    signal = np.array([])
    if isinstance(input_file, str):
        extension = os.path.splitext(input_file)[1].lower()
        if extension == ".wav":
            sampling_rate, signal = wavfile.read(input_file)
        else:
            try:
                audio = AudioSegment.from_file(input_file, extension)
                sampling_rate = audio.frame_rate
                if audio.sample_width == 2:
                    data = np.fromstring(audio._data, np.int16)
                elif audio.sample_width == 4:
                    data = np.fromstring(audio._data, np.int32)
                signal = []
                for ch in list(range(audio.channels)):
                    signal.append(data[ch::audio.channels])
                signal = np.array(signal).T
            except Exception as e:
                raise ValueError(
                    """
                    File extension not supported in {}.
                    Please convert your audio to .wav using pyAudioProcessing.convert_audio
                    using convert_files_to_wav method. Error {}
                    """.format(
                        input_file, e
                    )
                )
    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal
