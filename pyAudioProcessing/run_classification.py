#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:08:51 2019

@author: jsingh
"""
### Imports
import os
import warnings

from os import listdir
from os.path import isfile, join

from pyAudioProcessing.utils import write_to_json, ST_WIN, ST_STEP
from pyAudioProcessing.trainer import audioTrainTest as aT

warnings.simplefilter("ignore")


### Functions

def train_model(data_dirs, feature_names, classifier, classifier_name, use_file_names=False, file_names={}):
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
        ST_WIN,
        ST_STEP,
        classifier,
        classifier_name,
        #False,
        feats=feature_names,
        use_file_names=use_file_names,
        file_names=file_names
    )

def classify_data(data_dirs, feature_names, classifier, classifier_name, verbose=True, logfile=False, use_file_names=False, file_names={}):
    """
    Classify data in data_dirs
    by extracting features specified by feature_names
    and using the classifier saved by the name specified by classifier_name.
    """
    feature_names = [
        feat.lower().strip()
        for feat in feature_names
    ]
    if verbose:
        print(
            """\n Classifying using features {} with classifier {} that is saved as {}\n
            """.format(
                ", ".join(feature_names), classifier, classifier_name
            )
        )
    results = {}

    if use_file_names:
        master_fol = file_names
    else:
        master_fol = data_dirs

    for fol in master_fol:
        if verbose:
            print("\n", fol)
        results[fol] = {}
        if use_file_names:
            all_files = [
                f
                for f in master_fol[fol]
                if isfile(f) and f.endswith(".wav")
            ]
        else:
            all_files = [
                f
                for f in listdir(fol)
                if isfile(join(fol, f)) and f.endswith(".wav")
            ]
        correctly_classified = 0
        num_files = len(all_files)
        for f in all_files:
            if use_file_names:
                p = f
            else:
                p = join(fol, f)
            res = aT.fileClassification(
                p,
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
        if verbose:
            if correctly_classified == 0:
                print("Either you passed in data with unknown classes, or")
            print(
                "{} out of {} instances were classified correctly".format(
                    correctly_classified, num_files
                )
            )

    if logfile:
        write_to_json(
            logfile+"_"+classifier_name.split("/")[-1]+'.json',
            results
        )
    return results

def train_and_classify(
    folder_path,
    task,
    feature_names,
    classifier,
    classifier_name,
    logfile=False,
    use_file_names=False,
    file_names={}
):
    """
    Train on the data under folder_path or classify the data in folder path
    using features specified by feature_names and the specified classifier.
    """
    # Get all directories under folder_path
    if folder_path:
        data_dirs = [x[0] for x in os.walk(folder_path)][1:]
        print(
            "\n There are {} classes in the specified data folder\n".format(
                len(data_dirs)
            )
        )
    else:
        data_dirs=None
    if task == "train":
        train_model(data_dirs, feature_names, classifier, classifier_name, use_file_names=use_file_names, file_names=file_names)
    elif task == "classify":
        results = classify_data(data_dirs, feature_names, classifier, classifier_name, logfile=logfile, use_file_names=use_file_names, file_names=file_names)
        return results


def classify_pretrained(
    classifier_name,
    folder_path=None,
    file_names={}
):
    """
    Train on the data under folder_path or classify the data in folder path
    using features specified by feature_names and the specified classifier.
    """
    use_file_names = False
    if folder_path:
        # Get all directories under folder_path
        data_dirs = [x[0] for x in os.walk(folder_path)][1:]
    if file_names and len(file_names) > 0:
        data_dirs = None
        use_file_names = True
    if folder_path and (file_names and len(file_names) > 0):
        print(
            """
            \nCan't use two definitions of data paths.
            Using folder_path over file_names."""
        )
    if classifier_name == "speechVSmusic":
        feature_names = ["spectral", "chroma", "mfcc"]
        classifier_name = "models/speechVSmusic/svm_clf"
        classifier = "svm"
    elif classifier_name == "music genre":
        feature_names = ["gfcc", "spectral", "chroma", "mfcc"]
        classifier_name = "models/music genre/svm_clf"
        classifier = "svm"
    elif classifier_name == "speechVSmusicVSbirds":
        feature_names = ["spectral", "chroma", "mfcc"]
        classifier_name = "models/speechVSmusicVSbirds/svm_clf"
        classifier = "svm"
    else:
        raise("Classifier does not exist")
    return classify_data(
        data_dirs, feature_names, classifier, classifier_name,
        verbose=False, use_file_names=use_file_names, file_names=file_names
    )


def classify_ms(folder_path=None, file_names=None):
    return classify_pretrained("speechVSmusic", folder_path=folder_path, file_names=file_names)


def classify_msb(folder_path=None, file_names=None):
    return classify_pretrained("speechVSmusicVSbirds", folder_path=folder_path, file_names=file_names)


def classify_genre(folder_path=None, file_names=None):
    return classify_pretrained("music genre", folder_path=folder_path, file_names=file_names)

def train(folder_path=None, file_names=None, feature_names=["mfcc"], classifier="svm", classifier_name="svm_clf"):
    """
    Pass in either a path to the folder containing audio files in sub-folders
    as specified in the directory structure document; or pass in a dictionary
    of file_names containing class name as key, and path to each audio wav file
    in a list as values.
    Inputs:
    folder_path = "/Users/xyz/Documents/audio_files"
    # where the path specified contains sub-folders underneath, where each sub-folder
    represents a class and contains audio files for that class.
    OR,
    file_names = {
        "music" : ["/Users/xyz/Documents/audio/music1.wav", ...],
        "speech": ["/Users/xyz/Downloads/speech1.wav", ...],
        ...
    }
    """
    if folder_path:
        train_and_classify(
            folder_path,
            "train",
            feature_names,
            classifier,
            classifier_name,
            logfile=False,
            use_file_names=False,
            file_names={})
    if file_names:
        train_and_classify(
            None,
            "train",
            feature_names,
            classifier,
            classifier_name,
            logfile=False,
            use_file_names=True,
            file_names=file_names)


def classify(
    folder_path=None,
    file_names=None,
    feature_names=["mfcc"],
    classifier="svm",
    classifier_name="svm_clf",
    logfile=False
):
    """
    Pass in either a path to the folder containing audio files in sub-folders
    as specified in the directory structure document; or pass in a dictionary
    of file_names containing class name as key, and path to each audio wav file
    in a list as values.
    Inputs:
    folder_path = "/Users/xyz/Documents/audio_files"
    # where the path specified contains sub-folders underneath, where each sub-folder
    represents a class and contains audio files for that class.
    OR,
    file_names = {
        "music" : ["/Users/xyz/Documents/audio/music1.wav", ...],
        "speech": ["/Users/xyz/Downloads/speech1.wav", ...],
        ...
    }
    """
    if folder_path:
        return train_and_classify(
            folder_path,
            "classify",
            feature_names,
            classifier,
            classifier_name,
            logfile=logfile,
            use_file_names=False,
            file_names={})
    if file_names:
        return train_and_classify(
            None,
            "classify",
            feature_names,
            classifier,
            classifier_name,
            logfile=logfile,
            use_file_names=True,
            file_names=file_names)

if __name__ == "__main__":
    import argparse
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

    PARSER.add_argument(
        "-logfile", "--logfile", type=str, required=False,
        help="Path of file to log results in.",
        default="classifier_results"
    )

    ARGS = PARSER.parse_args()
    train_and_classify(
        ARGS.folder,
        ARGS.task,
        ARGS.feature_names,
        ARGS.classifier,
        ARGS.classifier_name,
        ARGS.logfile
    )
