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
import json

from trainer import audioTrainTest as aT

### Globals and Variables
PARSER = argparse.ArgumentParser(description="Run training or testing of audio samples.")
PARSER.add_argument(
    "-f", "--folder", type=str, required=True,
    help="Dir where data lives in folders names after classes."
)
PARSER.add_argument(
    "-t", "--task", type=str, required=True, choices=["train", "classify"],
    help="Train on data of classify data."
)
PARSER.add_argument(
    "-feats", "--feature-names", type=lambda s: [item for item in s.split(",")],
    default=["mfcc", "gfcc"],
    help="Features to compute.",
)
PARSER.add_argument(
    "-clf", "--classifier", type=str, required=True, help="Classifier to use or save.",
)
PARSER.add_argument(
    "-clfname", "--classifier-name", type=str, required=True,
    help="Name of the classifier to use or save.",
)


ARGS = PARSER.parse_args()
data_dirs = [x[0] for x in os.walk(ARGS.folder)][1:]
print("\n There are {} classes in the specified data folder\n".format(len(data_dirs)))

if ARGS.task == "train":
    print("""
        \n Training using features {} with classifier {} that will be saved as {}\n
        """.format(
            ", ".join(ARGS.feature_names), ARGS.classifier, ARGS.classifier_name)
    )
    aT.featureAndTrain(
        data_dirs,
        1.0, 1.0,
        aT.shortTermWindow, aT.shortTermStep,
        ARGS.classifier,
        ARGS.classifier_name,
        False,
        feats=ARGS.feature_names
    )

elif ARGS.task == "classify":
    print(
        """\n Classifying using features {} with classifier {} that is saved as {}\n
        """.format(
            ", ".join(ARGS.feature_names), ARGS.classifier, ARGS.classifier_name)
        )
    results = {}
    for fol in data_dirs:
        print("\n", fol)
        results[fol] = {}
        all_files = [f for f in listdir(fol) if isfile(join(fol, f)) and f.endswith(".wav")]
        correctly_classified = 0
        num_files = len(all_files)
        for f in all_files:
            res = aT.fileClassification(
                join(fol, f),
                ARGS.classifier_name,
                ARGS.classifier,
                feats=ARGS.feature_names
            )
            results[fol][f] = {
                "probabilities": list(res[1]),
                "classes": list(res[2])
            }
            indx = list(res[1]).index(max(res[1]))
            if res[2][indx] == fol.split("/")[-1]:
                correctly_classified += 1
        print(
            "{} out of {} instances were classified correctly".format(
                correctly_classified, num_files
            )
        )
    with open('classifier_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=1)
    print("\nResults saved in classifier_results.json\n")
