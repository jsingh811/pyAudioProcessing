#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:37:32 2021

@author: jsingh
"""
import os
import pytest
from pathlib import Path
from pyAudioProcessing.extract_features import get_features

def test_get_features():
    """
    Test get_features function
    """
    test_root = str(Path(__file__).parent.parent)
    data_dir = os.path.join(test_root, "data_samples/testing")
    features = get_features(data_dir, ["gfcc", "mfcc"])
    assert "speech" in features
    assert "music" in features
    assert len([i for i in features["speech"].keys() if "sleep.wav" in i]) > 0
    key = list(features["speech"].keys())[0]
    assert "features" in features["speech"][key]
    assert "feature_names" in features["speech"][key]
    assert  "mfcc_1_mean" in features["speech"][key]["feature_names"]
    assert  "gfcc_1_mean" in features["speech"][key]["feature_names"]
