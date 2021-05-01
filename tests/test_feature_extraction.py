#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:37:32 2021

@author: jsingh
"""
import pytest
from pyAudioProcessing.extract_features import get_features

def test_get_features():
    """
    Test get_features function
    """
    features = get_features("data_samples/testing", ["gfcc", "mfcc"])
    assert "musc" in features
    assert "speech" in features
    assert "data_samples/testing/speech/sleep.wav" in features["speech"]
    assert "features" in features["speech"]["data_samples/testing/speech/sleep.wav"]
    assert "feature_names" in features["speech"]["data_samples/testing/speech/sleep.wav"]
    assert  "mfcc_1_mean" in features["speech"]["data_samples/testing/speech/sleep.wav"]["feature_names"]
    assert  "gfcc_1_mean" in features["speech"]["data_samples/testing/speech/sleep.wav"]["feature_names"]
