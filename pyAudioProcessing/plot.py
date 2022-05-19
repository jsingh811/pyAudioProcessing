#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 18:22:15 2021

@author: jsingh
"""
# Imports

import numpy
import matplotlib.pyplot as plot
from pyAudioProcessing.utils import read_audio

# Functions

def spectrogram(
    input_file, show=True, save_to_disk=True, output_file="spectogram.png"
):
    """
    Plot spctrogram of input audio signal and save the output to disk.
    """
    sampling_rate, signal = read_audio(input_file)
    # Prepare the signal for spectrogram computation
    if len(signal) > 0:
        if len(numpy.array(signal[0], ndmin=1)) == 1:
            signal = [i for i in signal]
        elif len(numpy.array(signal[0], ndmin=1)) > 1:
            signal = [list(i)[0] for i in signal]
    # Plot the signal's spectrogram
    plot.title('Spectrogram')
    plot.specgram(signal, Fs=sampling_rate)
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    if save_to_disk:
        plot.savefig(output_file)
    if show:
        plot.show()

def time(input_file, show=True, save_to_disk=True, output_file="time.png"):
    """
    Plot time series audio amplitude and save the output to disk.
    """
    sampling_rate, signal = read_audio(input_file)
    plot.title('Time series plot')
    plot.plot(signal)

    plot.xlabel('Sample')

    plot.ylabel('Amplitude')
    if save_to_disk:
        plot.savefig(output_file)
    if show:
        plot.show()
