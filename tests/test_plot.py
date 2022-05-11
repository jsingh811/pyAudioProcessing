#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:37:32 2021

@author: jsingh
"""
import os
import pytest
from pathlib import Path
from pyAudioProcessing import plot

def test_spectrogram():
    """
    Test spectrogram function
    """
    test_root = str(Path(__file__).parent.parent)
    data_dir = os.path.join(test_root, "data_samples/testing")
    file_path = os.path.join(data_dir, "music/nearhou.wav")
    plot.spectrogram(file_path, show=False, save_to_disk=False)
