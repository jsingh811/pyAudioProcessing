#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:08:51 2019

@author: jsingh
"""

###############################################################################
# Imports
###############################################################################

from os import listdir, walk
from os.path import isfile, join, abspath, dirname

from pyAudioProcessing.utils import write_to_json, ST_WIN, ST_STEP
from pyAudioProcessing.trainer import audioTrainTest as aT


###############################################################################
# Functions
###############################################################################

def train_model(
    data_dirs,
    feature_names,
    classifier,
    classifier_name,
    use_file_names=False,
    file_names={},
):
    """
    Train classifier using data in data_dirs
    by extracting features specified by feature_names
    and saving the classifier as name specified by classifier_name.
    """
    feature_names = [feat.lower().strip() for feat in feature_names]
    print(
        """
        \n Training using features {} with classifier {} that will be saved as {}\n
        """.format(
            ", ".join(feature_names), classifier, classifier_name
        )
    )
    aT.featureAndTrain(
        data_dirs,
        1.0,
        1.0,
        ST_WIN,
        ST_STEP,
        classifier,
        classifier_name,
        # False,
        feats=feature_names,
        use_file_names=use_file_names,
        file_names=file_names,
    )


def classify_data(
    data_dirs,
    feature_names,
    classifier,
    classifier_name,
    verbose=True,
    logfile=False,
    use_file_names=False,
    file_names={},
):
    """
    Classify data in data_dirs
    by extracting features specified by feature_names
    and using the classifier saved by the name specified by classifier_name.
    """
    feature_names = [feat.lower().strip() for feat in feature_names]
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
            all_files = [f for f in master_fol[fol] if isfile(f) and f.endswith(".wav")]
        else:
            all_files = [
                f for f in listdir(fol) if isfile(join(fol, f)) and f.endswith(".wav")
            ]
        correctly_classified = 0
        num_files = len(all_files)
        for f in all_files:
            if use_file_names:
                p = f
            else:
                p = join(fol, f)
            res = aT.fileClassification(
                p, classifier_name, classifier, feats=feature_names
            )
            results[fol][f] = {"probabilities": list(res[1]), "classes": list(res[2])}
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
        write_to_json(logfile + "_" + classifier_name.split("/")[-1] + ".json", results)
    return results


def train_and_classify(
    folder_path,
    task,
    feature_names,
    classifier,
    classifier_name,
    logfile=False,
    use_file_names=False,
    file_names={},
):
    """
    Train on the data under folder_path or classify the data in folder path
    using features specified by feature_names and the specified classifier.
    """
    # Get all directories under folder_path
    if folder_path:
        data_dirs = [x[0] for x in walk(folder_path)][1:]
        print(
            "\n There are {} classes in the specified data folder\n".format(
                len(data_dirs)
            )
        )
    else:
        data_dirs = None
    if task == "train":
        train_model(
            data_dirs,
            feature_names,
            classifier,
            classifier_name,
            use_file_names=use_file_names,
            file_names=file_names,
        )
    elif task == "classify":
        results = classify_data(
            data_dirs,
            feature_names,
            classifier,
            classifier_name,
            logfile=logfile,
            use_file_names=use_file_names,
            file_names=file_names,
        )
        return results


def classify_pretrained(classifier_name, folder_path=None, file_names={}, file=None):
    """
    Classify data specified by input folder_path, file_names or single file
    using the pretrained classifier specified by input `classifier_name`.
    classifier_name can be one of the following:
            speechVSmusic
            speechVSmusicVSbirds
            music genre

    Pass in either a path to the folder containing audio files in sub-folders
    as specified in the directory structure document in the README;
    or pass in a dictionary `file_names` specifying audio paths as specified below;
    or a single file name specified by input `file`.
    in a list as values.
    Inputs:
        folder_path = "/Users/xyz/Documents/audio_files"
        # where the path specified contains sub-folders underneath, where each sub-folder
        contains audio files.
        OR,
        file_names = {
            "music" : ["/Users/xyz/Documents/audio/music1.wav", ...],
            "speech": ["/Users/xyz/Downloads/speech1.wav", ...],
            ...
        }
        OR,
        file = "/Users/xyz/Documents/audio.wav"

    """
    if folder_path and (file_names and len(file_names) > 0):
        print(
            """
            \nCan't use two definitions of data paths.
            Using file_names over folder_path."""
        )
    if file and folder_path:
        print(
            """
            \nCan't use two definitions of data paths.
            Using file over folder_path."""
        )
    if file and (file_names and len(file_names) > 0):
       print(
           """
           \nCan't use two definitions of data paths.
           Using file over file_names."""
       )

    use_file_names = False
    if folder_path:
        # Get all directories under folder_path
        data_dirs = [x[0] for x in walk(folder_path)][1:]
    if file_names and len(file_names) > 0:
        data_dirs = None
        use_file_names = True
    if file:
        data_dirs = None
        use_file_names = True
        file_names = {"audio__classification": [file]}

    if classifier_name == "speechVSmusic":
        feature_names = ["spectral", "chroma", "mfcc"]
        classifier_name = join(
            abspath(dirname(__file__)), "models/speechVSmusic/svm_clf"
        )
        classifier = "svm"
    elif classifier_name == "music genre":
        feature_names = ["gfcc", "spectral", "chroma", "mfcc"]
        classifier_name = join(
            abspath(dirname(__file__)), "models/music genre/svm_clf"
        )
        classifier = "svm"
    elif classifier_name == "speechVSmusicVSbirds":
        feature_names = ["spectral", "chroma", "mfcc"]
        classifier_name = join(
            abspath(dirname(__file__)), "models/speechVSmusicVSbirds/svm_clf"
        )
        classifier = "svm"
    else:
        raise ("Classifier does not exist")
    return classify_data(
        data_dirs,
        feature_names,
        classifier,
        classifier_name,
        verbose=False,
        use_file_names=use_file_names,
        file_names=file_names,
    )


def classify_ms(folder_path=None, file_names=None, file=None):
    """
    Pass in either a path to the folder containing audio files in sub-folders
    as specified in the directory structure document in the README;
    or pass in a dictionary `file_names` specifying audio paths as specified below;
    or a single file name specified by input `file`.
    in a list as values.
    Inputs:
        folder_path = "/Users/xyz/Documents/audio_files"
        # where the path specified contains sub-folders underneath, where each sub-folder
        contains audio files.
        OR,
        file_names = {
            "music" : ["/Users/xyz/Documents/audio/music1.wav", ...],
            "speech": ["/Users/xyz/Downloads/speech1.wav", ...],
            ...
        }
        OR,
        file = "/Users/xyz/Documents/audio.wav"
    """
    return classify_pretrained(
        "speechVSmusic", folder_path=folder_path, file_names=file_names, file=file
    )


def classify_msb(folder_path=None, file_names=None, file=None):
    """
    Pass in either a path to the folder containing audio files in sub-folders
    as specified in the directory structure document in the README;
    or pass in a dictionary `file_names` specifying audio paths as specified below;
    or a single file name specified by input `file`.
    in a list as values.
    Inputs:
        folder_path = "/Users/xyz/Documents/audio_files"
        # where the path specified contains sub-folders underneath, where each sub-folder
        contains audio files.
        OR,
        file_names = {
            "music" : ["/Users/xyz/Documents/audio/music1.wav", ...],
            "speech": ["/Users/xyz/Downloads/speech1.wav", ...],
            ...
        }
        OR,
        file = "/Users/xyz/Documents/audio.wav"
    """
    return classify_pretrained(
        "speechVSmusicVSbirds", folder_path=folder_path, file_names=file_names, file=file
    )


def classify_genre(folder_path=None, file_names=None, file=None):
    """
    Pass in either a path to the folder containing audio files in sub-folders
    as specified in the directory structure document in the README;
    or pass in a dictionary `file_names` specifying audio paths as specified below;
    or a single file name specified by input `file`.
    in a list as values.
    Inputs:
        folder_path = "/Users/xyz/Documents/audio_files"
        # where the path specified contains sub-folders underneath, where each sub-folder
        contains audio files.
        OR,
        file_names = {
            "music" : ["/Users/xyz/Documents/audio/music1.wav", ...],
            "speech": ["/Users/xyz/Downloads/speech1.wav", ...],
            ...
        }
        OR,
        file = "/Users/xyz/Documents/audio.wav"
    """
    return classify_pretrained(
        "music genre", folder_path=folder_path, file_names=file_names, file=file
    )


def train(
    folder_path=None,
    file_names=None,
    feature_names=["mfcc"],
    classifier="svm",
    classifier_name="svm_clf",
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

        classifier
            Classifier. Choose from svm, svm_rbf, randomforest, logisticregression,
            knn, gradientboosting, extratrees.
        classifier_name
            Path to the saved classifier model.
        feature_names
            defaults to ["mfcc"]. Choose from "mfcc", "gfcc", "spectral", "chroma".

    The model is saved in the specified classifier_name path .
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
            file_names={},
        )
    if file_names:
        train_and_classify(
            None,
            "train",
            feature_names,
            classifier,
            classifier_name,
            logfile=False,
            use_file_names=True,
            file_names=file_names,
        )


def classify(
    folder_path=None,
    file_names=None,
    feature_names=["mfcc"],
    classifier="svm",
    classifier_name="svm_clf",
    file=None,
    logfile=False,
):
    """
    Pass in either a path to the folder containing audio files in sub-folders
    as specified in the directory structure document; or pass in a dictionary
    of file_names containing class name as key, and path to each audio wav file
    in a list as values; or a single file path.
    Inputs:
        folder_path = "/Users/xyz/Documents/audio_files"
        # where the path specified contains sub-folders underneath, where each sub-folder
        represents a class and contains audio files for that class.
        OR,
        file_names = {
            "music" : ["/Users/xyz/Documents/audio/music1.wav", ...],
            "speech": ["/Users/xyz/Downloads/speech1.wav", ...],
            ...
        },
        OR
        file = "/Users/xyz/Documents/audio.wav"

        classifier
            Classifier. Choose from svm, svm_rbf, randomforest, logisticregression,
            knn, gradientboosting, extratrees.
            Must be same features selected during training the classifier
        classifier_name
            Path to the saved classifier model.
        feature_names
            defaults to ["mfcc"]. Choose from "mfcc", "gfcc", "spectral", "chroma".
            Must be same features selected during training the classifier
        logfile
            If you want to log the classification results in a file json

    Returns
        Classification results as a dictionary of the form
        {
            "music" : {
                "/Users/xyz/Documents/audio/music1.wav": [
                    [<classification probabilities>], [<class names>]
                ], ...
            }
            "speech": {"/Users/xyz/Downloads/speech1.wav": [
                    [<classification probabilities>], [<class names>]
                ], ...
            }
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
            file_names={},
        )
    if file_names:
        return train_and_classify(
            None,
            "classify",
            feature_names,
            classifier,
            classifier_name,
            logfile=logfile,
            use_file_names=True,
            file_names=file_names,
        )
    if file:
        return train_and_classify(
            None,
            "classify",
            feature_names,
            classifier,
            classifier_name,
            logfile=logfile,
            use_file_names=True,
            file_names={"audio__classification": [file]},
        )


if __name__ == "__main__":
    import argparse

    ### Globals and Variables
    PARSER = argparse.ArgumentParser(
        description="Run training or testing of audio samples."
    )
    PARSER.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Dir where data lives in folders names after classes.",
    )
    PARSER.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        choices=["train", "classify"],
        help="Train on data of classify data.",
    )
    PARSER.add_argument(
        "-feats",
        "--feature-names",
        type=lambda s: [item for item in s.split(",")],
        default=["mfcc", "gfcc", "chroma", "spectral"],
        help="Features to compute.",
    )
    PARSER.add_argument(
        "-clf",
        "--classifier",
        type=str,
        required=True,
        help="Classifier to use or save.",
    )
    PARSER.add_argument(
        "-clfname",
        "--classifier-name",
        type=str,
        required=True,
        help="Name of the classifier to use or save.",
    )

    PARSER.add_argument(
        "-logfile",
        "--logfile",
        type=str,
        required=False,
        help="Path of file to log results in.",
        default="classifier_results",
    )

    ARGS = PARSER.parse_args()
    train_and_classify(
        ARGS.folder,
        ARGS.task,
        ARGS.feature_names,
        ARGS.classifier,
        ARGS.classifier_name,
        ARGS.logfile,
    )
