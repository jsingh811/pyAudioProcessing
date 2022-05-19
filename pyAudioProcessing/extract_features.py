#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:55:23 2021

@author: jsingh
"""

###############################################################################
# Imports
###############################################################################

import os

from pyAudioProcessing.utils import ST_WIN, ST_STEP
from pyAudioProcessing.trainer import audioTrainTest as aT


###############################################################################
# Functions
###############################################################################


def get_features(
        folder_path=None, 
        feature_names=["mfcc", "gfcc"], 
        file_names={}, 
        file=None
):
    """
    Extracts features specified in feature_names for 
    1. audios in every folder inside `folder_path`, OR
    2. audios in every path specified by `file_names`
        {
            "music": [<path to audio 1>, <path of audio 2>, ... ],
            ...
        }
    3. a single audio file path specified by `file`
    
    Inputs:
        folder_path
        file_names
        file
        feature_names:
            Choose from gfcc, mfcc, spectral, chroma
    Returns a dict of the format
    {
        <every folder name inside folder_path>: {
            <file name> :{
                "features": list, "feature_names": list
            },
        .. },
    ..}
    """
    use_file_names = False
    if folder_path:
        data_dirs = [x[0] for x in os.walk(folder_path)][1:]
    if file_names and len(file_names) > 0:
        data_dirs = None
        use_file_names = True
    if file:
        data_dirs = None
        use_file_names = True
        file_names = {"audio": [file]}
    feature_names = [feat.lower().strip() for feat in feature_names]
    print(
        """
        \n Extracting features {} \n
        """.format(
            ", ".join(feature_names)
        )
    )
    features, class_names, file_names, feat_names = aT.extract_features(
        data_dirs,
        1.0,
        1.0,
        ST_WIN,
        ST_STEP,
        feature_names,
        use_file_names=use_file_names,
        file_names=file_names,
    )

    class_file_feats = {}
    for inx in range(len(class_names)):
        files = file_names[inx]
        class_file_feats[class_names[inx]] = {}
        for sub_inx in range(len(files)):
            class_file_feats[class_names[inx]][files[sub_inx]] = {
                "features": list(features[inx][sub_inx]),
                "feature_names": feat_names[inx],
            }

    return class_file_feats


if __name__ == "__main__":
    import argparse
    from pyAudioProcessing.utils import write_to_json

    PARSER = argparse.ArgumentParser(description="Extract features from audio samples.")
    PARSER.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Dir where data lives in folders names after classes.",
    )
    PARSER.add_argument(
        "-feats",
        "--feature-names",
        type=lambda s: [item for item in s.split(",")],
        default=["mfcc", "gfcc", "chroma", "spectral"],
        help="Features to compute.",
    )
    ARGS = PARSER.parse_args()
    # dict with structure
    # {
    #    <folder name inside ARGS.folder>: {
    #       <file name> :{
    #           "features": list, "feature_names": list
    #       }, .. }, ..}
    file_features = get_features(
        folder_path=ARGS.folder,
        feature_names=ARGS.feature_names
    )
    write_to_json("audio_features.json", file_features)
