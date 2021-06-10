#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:08:51 2019

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
    description="Run training or testing of audio samples."
)
PARSER.add_argument(
    "-f", "--folder", type=str, required=True,
    help="Dir where data lives in folders names after classes."
)
PARSER.add_argument(
    "-t", "--task", type=str, required=True, choices=["train", "classify"],
    help="Train on data of classify data."
)
PARSER.add_argument(
    "-feats", "--feature-names",
    type=lambda s: [item for item in s.split(",")],
    default=["mfcc", "gfcc", "chroma", "spectral"],
    help="Features to compute.",
)
PARSER.add_argument(
    "-clf", "--classifier", type=str, required=True,
    help="Classifier to use or save.",
)
PARSER.add_argument(
    "-clfname", "--classifier-name", type=str, required=True,
    help="Name of the classifier to use or save.",
)


### Functions

def train_model(data_dirs, feature_names, classifier, classifier_name):
    """
    Train classifier using data in data_dirs
    by extracting features specified by feature_names
    and saving the classifier as name specified by classifier_name.
    """
    feature_names = [
        feat.lower().strip()
        for feat in feature_names
    ]
    print("""
        \n Training using features {} with classifier {} that will be saved as {}\n
        """.format(
            ", ".join(feature_names), classifier, classifier_name)
    )
    aT.featureAndTrain(
        data_dirs,
        1.0, 1.0,
        aT.shortTermWindow,
        aT.shortTermStep,
        classifier,
        classifier_name,
        False,
        feats=feature_names
    )

def classify_data(data_dirs, feature_names, classifier, classifier_name):
    """
    Classify data in data_dirs
    by extracting features specified by feature_names
    and using the classifier saved by the name specified by classifier_name.
    """
    feature_names = [
        feat.lower().strip()
        for feat in feature_names
    ]
    print(
        """\n Classifying using features {} with classifier {} that is saved as {}\n
        """.format(
            ", ".join(feature_names), classifier, classifier_name
        )
    )
    results = {}
    for fol in data_dirs:
        print("\n", fol)
        results[fol] = {}
        all_files = [
            f
            for f in listdir(fol)
            if isfile(join(fol, f)) and f.endswith(".wav")
        ]
        correctly_classified = 0
        num_files = len(all_files)
        for f in all_files:
            res = aT.fileClassification(
                join(fol, f),
                classifier_name,
                classifier,
                feats=feature_names
            )
            results[fol][f] = {
                "probabilities": list(res[1]),
                "classes": list(res[2])
            }
            indx = list(res[1]).index(max(res[1]))
            if res[2][indx] == fol.split("/")[-1]:
                correctly_classified += 1
        if correctly_classified == 0:
            print("Either you passed in data with unknown classes, or")
        print(
            "{} out of {} instances were classified correctly".format(
                correctly_classified, num_files
            )
        )
    write_to_json('classifier_results.json', results)

def train_and_classify(
    folder_path,
    task,
    feature_names,
    classifier,
    classifier_name
):
    """
    Train on the data under folder_path or classify the data in folder path
    using features specified by feature_names and the specified classifier.
    """
    # Get all directories under folder_path
    data_dirs = [x[0] for x in os.walk(folder_path)][1:]
    print(
        "\n There are {} classes in the specified data folder\n".format(
            len(data_dirs)
        )
    )
    if task == "train":
        train_model(data_dirs, feature_names, classifier, classifier_name)
    elif task == "classify":
        classify_data(data_dirs, feature_names, classifier, classifier_name)


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    train_and_classify(
        ARGS.folder,
        ARGS.task,
        ARGS.feature_names,
        ARGS.classifier,
        ARGS.classifier_name
    )
