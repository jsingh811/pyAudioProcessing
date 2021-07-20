#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:27:32 2021

@author: jsingh
"""
### Imports
import json
import os
import numpy
from scipy.io import wavfile

### Functions
def write_to_json(file_name, data):
    """
    Write data to file_name.
    """
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile, indent=1)
    print("\nResults saved in {}\n".format(file_name))

def read_audio(input_file):
    """
    Reads input audio file and returns sampling frequency along with the signal.
    """
    sampling_rate = 0
    signal = numpy.array([])
    if isinstance(input_file, str):
        extension = os.path.splitext(input_file)[1].lower()
        if extension == '.wav':
            sampling_rate, signal = wavfile.read(input_file)
        else:
            raise ValueError("""
                File extension not supported.
                Please convert your audio to .wav using pyAudioProcessing.convert_audio
                using convert_files_to_wav method.
            """) 
    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal
