#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:23:55 2021

@author: jsingh
"""
# Imports
import numpy as np
from scipy.io import wavfile
from pyAudioProcessing.utils import read_audio, convert_audio_to_mono

# Globals
LOW_PT_THRESHOLD = 0.01

# Functions


def remove_silence(input_file, output_file="clean.wav", thr=LOW_PT_THRESHOLD):
    """
    Remove silences from input audio file.
    Works for both stereo and mono type of audios.
    TODO: make more efficient.
    """
    sampling_rate, signal = read_audio(input_file)
    if len(np.array(signal[0], ndmin=1)) > 1:
        signal = np.array([list(i) for i in signal if not np.all((i == 0))])
        mean_signal = convert_audio_to_mono(signal)
        thrs_p = thr * max(mean_signal)
        thrs_n = thr * min(mean_signal)
        signal = np.array(
            [list(i) for i in signal if np.all((i > thrs_p)) or np.all((i < thrs_n))]
        )
    else:
        thrs_p = thr * max(signal)
        thrs_n = thr * min(signal)
        signal = np.array([i for i in signal if i > thrs_p or i < thrs_n])
    wavfile.write(output_file, sampling_rate, signal)
