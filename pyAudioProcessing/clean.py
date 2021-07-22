#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:23:55 2021

@author: jsingh
"""
# Imports
import os
import numpy
from scipy.io import wavfile
from pyAudioProcessing.utils import read_audio

# Globals
LOW_PT_THRESHOLD = 0.01

# Functions

def remove_silence(
    input_file, output_file="without_sil.wav", thr=LOW_PT_THRESHOLD
):
    """
    Remove silences from input audio file.
    """
    sampling_rate, signal = read_audio(input_file)
    
    if len(numpy.array(signal[0], ndmin=1)) > 1:
        signal = numpy.array(
            [
                list(i)
                for i in signal
                if not numpy.all((i == 0))
            ]
        )
        mean_signal = [numpy.mean(list(i)) for i in signal]
        thrs_p = thr * max(mean_signal)
        thrs_n = thr * min(mean_signal)
        signal = numpy.array(
            [
                list(i)
                for i in signal
                if numpy.all((i > thrs_p)) or numpy.all((i < thrs_n))
            ]
        )
    else:
        thrs_p = thr * max(signal)
        thrs_n = thr * min(signal)
        signal = numpy.array(
            [
                i
                for i in signal
                if i > thrs_p or i < thrs_n
            ]
        )
    wavfile.write(output_file, sampling_rate, signal)


