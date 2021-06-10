#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:55:23 2021

@author: jsingh
"""
### Imports
import argparse
import os
from os import listdir
from os.path import isfile, join

from pyAudioProcessing.utils import write_to_json
from pyAudioProcessing.trainer import audioTrainTest as aT


### Globals and Variables
PARSER = argparse.ArgumentParser(
    description="Extract features from audio samples."
)
PARSER.add_argument(
    "-f", "--folder", type=str, required=True,
    help="Dir where data lives in folders names after classes."
)
PARSER.add_argument(
    "-feats", "--feature-names",
    type=lambda s: [item for item in s.split(",")],
    default=["mfcc", "gfcc", "chroma", "spectral"],
    help="Features to compute.",
)

### Functions

def get_features(folder_path, feature_names):
    """
    Extracts features specified in feature_names for every folder inside folder_path.
    Returns a dict of the format
    {
        <every folder name inside folder_path>: {
            <file name> :{
                "features": list, "feature_names": list
            },
        .. },
    ..}
    """
    data_dirs = [x[0] for x in os.walk(folder_path)][1:]
    feature_names = [feat.lower().strip() for feat in feature_names]
    print("""
        \n Extracting features {} \n
        """.format(
            ", ".join(
                feature_names
            )
        )
    )
    features, class_names, file_names, feat_names = aT.extract_features(
        data_dirs,
        1.0, 1.0,
        aT.shortTermWindow,
        aT.shortTermStep,
        False,
        feature_names
    )

    class_file_feats = {}
    for inx in range(len(class_names)):
        files = file_names[inx]
        class_file_feats[class_names[inx]] = {}
        for sub_inx in range(len(files)):
            class_file_feats[class_names[inx]][files[sub_inx]] = {
                "features": list(features[inx][sub_inx]),
                "feature_names": feat_names[inx]
            }

    return class_file_feats


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    # dict with structure
    # {
    #    <folder name inside ARGS.folder>: {
    #       <file name> :{
    #           "features": list, "feature_names": list
    #       }, .. }, ..}
    file_features = get_features(
        ARGS.folder,
        ARGS.feature_names
    )
    write_to_json('audio_features.json', file_features)
