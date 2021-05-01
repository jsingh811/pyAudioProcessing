### This script is derived from https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioTrainTest.py)
import sys
import numpy
import os
import glob
import pickle as cPickle
import signal
import csv
import ntpath
import sklearn
import pyAudioProcessing.features.audioFeatureExtraction as aF
from pyAudioAnalysis.audioTrainTest import (
    writeTrainDataToARFF, normalizeFeatures, trainSVM,
    trainSVM_RBF, trainRandomForest, trainGradientBoosting, trainExtraTrees,
    listOfFeatures2Matrix, evaluateRegression, trainSVMregression,
    trainSVMregression_rbf, trainRandomForestRegression, load_model_knn,
    load_model, regressionWrapper, randSplitFeatures,
    listOfFeatures2Matrix, normalizeFeatures, randSplitFeatures,
    printConfusionMatrix, trainKNN, classifierWrapper
)
from pyAudioAnalysis import audioBasicIO

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! - EXIT')
    os.system("stty -cbreak echo")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

shortTermWindow = 0.050
shortTermStep = 0.050
eps = 0.00000001

def classifierWrapperHead(classifier, classifier_type, test_sample):
    '''
    '''
    if classifier_type == "logisticregression":
        R = classifier.predict(test_sample.reshape(1,-1))[0]
        P = classifier.predict_proba(test_sample.reshape(1,-1))[0]
        return [R, P]
    else:
        return classifierWrapper(classifier, classifier_type, test_sample)

def trainLogisticRegression(features, Cparam):
    '''
    Train a multi-class probabilitistic Logistic Regression classifier.
    Note:     This function is simply a wrapper to the sklearn functionality for logistic regression training
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements contain numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - Cparam:           Logistic Regression parameter C (Inverse of regularization strength)
    RETURNS:
        - lr:              the trained logistic regression variable
    NOTE:
        This function trains a Logistic Regression model for a given C value.
        For a different kernel, other types of parameters should be provided.
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    lr = sklearn.linear_model.LogisticRegression(C=Cparam, multi_class="ovr")
    lr.fit(X,Y)

    return lr

def evaluateclassifier(features, class_names, n_exp, classifier_name, Params, parameterMode, perTrain=0.90):
    '''
    ARGUMENTS:
        features:     a list ([numOfClasses x 1]) whose elements contain numpy matrices of features.
                each matrix features[i] of class i is [n_samples x numOfDimensions]
        class_names:    list of class names (strings)
        n_exp:        number of cross-validation experiments
        classifier_name: svm or knn or randomforest
        Params:        list of classifier parameters (for parameter tuning during cross-validation)
        parameterMode:    0: choose parameters that lead to maximum overall classification ACCURACY
                1: choose parameters that lead to maximum overall f1 MEASURE
    RETURNS:
         bestParam:    the value of the input parameter that optimizes the selected performance measure
    '''

    # feature normalization:
    (features_norm, MEAN, STD) = normalizeFeatures(features)
    #features_norm = features;
    n_classes = len(features)
    ac_all = []
    f1_all = []
    precision_classes_all = []
    recall_classes_all = []
    f1_classes_all = []
    cms_all = []

    # compute total number of samples:
    n_samples_total = 0
    for f in features:
        n_samples_total += f.shape[0]
    if n_samples_total > 1000 and n_exp > 50:
        n_exp = 50
        print("Number of training experiments changed to 50 due to high number of samples")
    if n_samples_total > 2000 and n_exp > 10:
        n_exp = 10
        print("Number of training experiments changed to 10 due to high number of samples")

    for Ci, C in enumerate(Params):
        # for each param value
        cm = numpy.zeros((n_classes, n_classes))
        for e in range(n_exp):
            # for each cross-validation iteration:
            print("Param = {0:.5f} - classifier Evaluation "
                  "Experiment {1:d} of {2:d}".format(C, e+1, n_exp))
            # split features:
            f_train, f_test = randSplitFeatures(features_norm, perTrain)
            # train multi-class svms:
            if classifier_name == "svm":
                classifier = trainSVM(f_train, C)
            elif classifier_name == "svm_rbf":
                classifier = trainSVM_RBF(f_train, C)
            elif classifier_name == "knn":
                classifier = trainKNN(f_train, C)
            elif classifier_name == "randomforest":
                classifier = trainRandomForest(f_train, C)
            elif classifier_name == "gradientboosting":
                classifier = trainGradientBoosting(f_train, C)
            elif classifier_name == "extratrees":
                classifier = trainExtraTrees(f_train, C)
            elif classifier_name == "logisticregression":
                classifier = trainLogisticRegression(f_train, C)

            cmt = numpy.zeros((n_classes, n_classes))
            for c1 in range(n_classes):
                n_test_samples = len(f_test[c1])
                res = numpy.zeros((n_test_samples, 1))
                for ss in range(n_test_samples):
                    [res[ss], _] = classifierWrapperHead(classifier,
                                                     classifier_name,
                                                     f_test[c1][ss])
                for c2 in range(n_classes):
                    cmt[c1][c2] = float(len(numpy.nonzero(res == c2)[0]))
            cm = cm + cmt
        cm = cm + 0.0000000010
        rec = numpy.zeros((cm.shape[0], ))
        pre = numpy.zeros((cm.shape[0], ))

        for ci in range(cm.shape[0]):
            rec[ci] = cm[ci, ci] / numpy.sum(cm[ci, :])
            pre[ci] = cm[ci, ci] / numpy.sum(cm[:, ci])
        precision_classes_all.append(pre)
        recall_classes_all.append(rec)
        f1 = 2 * rec * pre / (rec + pre)
        f1_classes_all.append(f1)
        ac_all.append(numpy.sum(numpy.diagonal(cm)) / numpy.sum(cm))

        cms_all.append(cm)
        f1_all.append(numpy.mean(f1))

    print("\t\t", end="")
    for i, c in enumerate(class_names):
        if i == len(class_names)-1:
            print("{0:s}\t\t".format(c), end="")
        else:
            print("{0:s}\t\t\t".format(c), end="")
    print("OVERALL")
    print("\tC", end="")
    for c in class_names:
        print("\tPRE\tREC\tf1", end="")
    print("\t{0:s}\t{1:s}".format("ACC", "f1"))
    best_ac_ind = numpy.argmax(ac_all)
    best_f1_ind = numpy.argmax(f1_all)
    for i in range(len(precision_classes_all)):
        print("\t{0:.3f}".format(Params[i]), end="")
        for c in range(len(precision_classes_all[i])):
            print("\t{0:.1f}\t{1:.1f}\t{2:.1f}".format(100.0 * precision_classes_all[i][c],
                                                       100.0 * recall_classes_all[i][c],
                                                       100.0 * f1_classes_all[i][c]), end="")
        print("\t{0:.1f}\t{1:.1f}".format(100.0 * ac_all[i], 100.0 * f1_all[i]), end="")
        if i == best_f1_ind:
            print("\t best f1", end="")
        if i == best_ac_ind:
            print("\t best Acc", end="")
        print("")

    if parameterMode == 0:    # keep parameters that maximize overall classification accuracy:
        print("Confusion Matrix:")
        printConfusionMatrix(cms_all[best_ac_ind], class_names)
        return Params[best_ac_ind]
    elif parameterMode == 1:  # keep parameters that maximize overall f1 measure:
        print("Confusion Matrix:")
        printConfusionMatrix(cms_all[best_f1_ind], class_names)
        return Params[best_f1_ind]

def extract_raw_features(
    list_of_dirs, mt_win, mt_step, st_win, st_step, compute_beat, feats
):
    """
    Extracts raw features specified by feats.
    Returns features, class names, file names and feature names.
    """
    [features, classNames, fileNames, featureNames] = aF.dirsWavFeatureExtraction(
        list_of_dirs,
        mt_win,
        mt_step,
        st_win,
        st_step,
        compute_beat=compute_beat,
        feats=feats
    )
    return features, classNames, fileNames, featureNames

def format_features(features):
    """
    Formats input list of features.
    """
    formatted_features = []
    for f in features:
        fTemp = []
        for i in range(f.shape[0]):
            temp = f[i,:]
            if (not numpy.isnan(temp).any()) and (not numpy.isinf(temp).any()):
                fTemp.append(temp.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        formatted_features.append(numpy.array(fTemp))
    return formatted_features

def extract_features(
    list_of_dirs, mt_win, mt_step, st_win, st_step, compute_beat, feats
):
    """
    Extracts features and returns features, class names, file names
    and feature names.
    """
    features, classNames, fileNames, featureNames = extract_raw_features(
        list_of_dirs, mt_win, mt_step, st_win, st_step, compute_beat, feats
    )
    features = format_features(features)
    return features, classNames, fileNames, featureNames

def featureAndTrain(list_of_dirs, mt_win, mt_step, st_win, st_step,
                    classifier_type, model_name,
                    compute_beat=False, perTrain=0.90, feats=["gfcc", "mfcc", "spectral", "chroma"]):
    '''
    This function is used as a wrapper to segment-based audio feature extraction and classifier training.
    ARGUMENTS:
        list_of_dirs:        list of paths of directories. Each directory contains a signle audio class whose samples are stored in separate WAV files.
        mt_win, mt_step:        mid-term window length and step
        st_win, st_step:        short-term window and step
        classifier_type:        "svm" or "knn" or "randomforest" or "gradientboosting" or "extratrees"
        model_name:        name of the model to be saved
    RETURNS:
        None. Resulting classifier along with the respective model parameters are saved on files.
    '''

    # STEP A: Feature Extraction:
    features, classNames, _, _ = extract_raw_features(list_of_dirs,
                                            mt_win,
                                            mt_step,
                                            st_win,
                                            st_step,
                                            compute_beat=compute_beat,
                                            feats=feats)

    if len(features) == 0:
        print("trainSVM_feature ERROR: No data found in any input folder!")
        return

    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    writeTrainDataToARFF(model_name, features, classNames, feature_names)

    for i, f in enumerate(features):
        if len(f) == 0:
            print("trainSVM_feature ERROR: " + list_of_dirs[i] + " folder is empty or non-existing!")
            return

    # STEP B: classifier Evaluation and Parameter Selection:
    if classifier_type == "svm" or classifier_type == "svm_rbf":
        classifier_par = numpy.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0])
    elif classifier_type == "randomforest":
        classifier_par = numpy.array([10, 25, 50, 100,200,500])
    elif classifier_type == "knn":
        classifier_par = numpy.array([1, 3, 5, 7, 9, 11, 13, 15])
    elif classifier_type == "gradientboosting":
        classifier_par = numpy.array([10, 25, 50, 100,200,500])
    elif classifier_type == "extratrees":
        classifier_par = numpy.array([10, 25, 50, 100,200,500])
    elif classifier_type == "logisticregression":
        classifier_par = numpy.array([0.01, 0.1, 1, 5])

    # get optimal classifier parameter
    features = format_features(features)

    bestParam = evaluateclassifier(
        features, classNames, 100, classifier_type, classifier_par, 0, perTrain
    )

    print("Selected params: {0:.5f}".format(bestParam))

    C = len(classNames)
    [features_norm, MEAN, STD] = normalizeFeatures(features) # normalize features
    MEAN = MEAN.tolist()
    STD = STD.tolist()
    featuresNew = features_norm

    # STEP C: Save the classifier to file
    if classifier_type == "svm":
        classifier = trainSVM(featuresNew, bestParam)
    elif classifier_type == "svm_rbf":
        classifier = trainSVM_RBF(featuresNew, bestParam)
    elif classifier_type == "randomforest":
        classifier = trainRandomForest(featuresNew, bestParam)
    elif classifier_type == "gradientboosting":
        classifier = trainGradientBoosting(featuresNew, bestParam)
    elif classifier_type == "extratrees":
        classifier = trainExtraTrees(featuresNew, bestParam)
    elif classifier_type == "logisticregression":
        classifier = trainLogisticRegression(featuresNew, bestParam)

    if classifier_type == "knn":
        [X, Y] = listOfFeatures2Matrix(featuresNew)
        X = X.tolist()
        Y = Y.tolist()
        fo = open(model_name, "wb")
        cPickle.dump(X, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(Y,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(bestParam,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mt_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mt_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(st_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(st_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(compute_beat, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifier_type == "svm" or classifier_type == "svm_rbf" or \
                    classifier_type == "randomforest" or \
                    classifier_type == "gradientboosting" or \
                    classifier_type == "extratrees" or \
                    classifier_type == "logisticregression":
        with open(model_name, 'wb') as fid:
            cPickle.dump(classifier, fid)
        fo = open(model_name + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mt_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mt_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(st_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(st_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(compute_beat, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()


def featureAndTrainRegression(dir_name, mt_win, mt_step, st_win, st_step,
                              model_type, model_name, compute_beat=False,
                              feats=["gfcc", "mfcc"]):
    '''
    This function is used as a wrapper to segment-based audio feature extraction and classifier training.
    ARGUMENTS:
        dir_name:        path of directory containing the WAV files and Regression CSVs
        mt_win, mt_step:        mid-term window length and step
        st_win, st_step:        short-term window and step
        model_type:        "svm" or "knn" or "randomforest"
        model_name:        name of the model to be saved
    RETURNS:
        None. Resulting regression model along with the respective model parameters are saved on files.
    '''
    # STEP A: Feature Extraction:
    [features, _, filenames] = aF.dirsWavFeatureExtraction([dir_name],
                                                           mt_win,
                                                           mt_step,
                                                           st_win,
                                                           st_step,
                                                           compute_beat=compute_beat,
                                                           feats=feats)
    features = features[0]
    filenames = [ntpath.basename(f) for f in filenames[0]]
    f_final = []

    # Read CSVs:
    CSVs = glob.glob(dir_name + os.sep + "*.csv")
    regression_labels = []
    regression_names = []
    f_final = []
    for c in CSVs:                                                            # for each CSV
        cur_regression_labels = []
        f_temp = []
        with open(c, 'rt') as csvfile:                                        # open the csv file that contains the current target value's annotations
            CSVreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in CSVreader:
                if len(row) == 2:                                             # if the current row contains two fields (filename, target value)
                    if row[0] in filenames:                                   # ... and if the current filename exists in the list of filenames
                        index = filenames.index(row[0])
                        cur_regression_labels.append(float(row[1]))
                        f_temp.append(features[index,:])
                    else:
                        print("Warning: {} not found in list of files.".format(row[0]))
                else:
                    print("Warning: Row with unknown format in regression file")

        f_final.append(numpy.array(f_temp))
        regression_labels.append(numpy.array(cur_regression_labels))                          # cur_regression_labels is the list of values for the current regression problem
        regression_names.append(ntpath.basename(c).replace(".csv", ""))        # regression task name
        if len(features) == 0:
            print("ERROR: No data found in any input folder!")
            return

    n_feats = f_final[0].shape[1]

    # TODO: ARRF WRITE????
    # STEP B: classifier Evaluation and Parameter Selection:
    if model_type == "svm" or model_type == "svm_rbf":
        model_params = numpy.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0])
    elif model_type == "randomforest":
        model_params = numpy.array([5, 10, 25, 50, 100])

#    elif model_type == "knn":
#        model_params = numpy.array([1, 3, 5, 7, 9, 11, 13, 15]);
    errors = []
    errors_base = []
    best_params = []

    for iRegression, r in enumerate(regression_names):
        # get optimal classifeir parameter:
        print("Regression task " + r)
        bestParam, error, berror = evaluateRegression(f_final[iRegression],
                                                      regression_labels[iRegression],
                                                      100, model_type,
                                                      model_params)
        errors.append(error)
        errors_base.append(berror)
        best_params.append(bestParam)
        print("Selected params: {0:.5f}".format(bestParam))

        [features_norm, MEAN, STD] = normalizeFeatures([f_final[iRegression]])        # normalize features

        # STEP C: Save the model to file
        if model_type == "svm":
            classifier, _ = trainSVMregression(features_norm[0],
                                               regression_labels[iRegression],
                                               bestParam)
        if model_type == "svm_rbf":
            classifier, _ = trainSVMregression_rbf(features_norm[0],
                                                   regression_labels[iRegression],
                                                   bestParam)
        if model_type == "randomforest":
            classifier, _ = trainRandomForestRegression(features_norm[0],
                                                        regression_labels[iRegression],
                                                        bestParam)

        if model_type == "svm" or model_type == "svm_rbf" or model_type == "randomforest":
            with open(model_name + "_" + r, 'wb') as fid:
                cPickle.dump(classifier, fid)
            fo = open(model_name + "_" + r + "MEANS", "wb")
            cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(STD,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(mt_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(mt_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(st_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(st_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(compute_beat, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            fo.close()
    return errors, errors_base, best_params



def fileClassification(inputFile, model_name, model_type, feats=["gfcc", "mfcc"]):
    # Load classifier:

    if not os.path.isfile(model_name):
        print("fileClassification: input model_name not found!")
        return (-1, -1, -1)

    if not os.path.isfile(inputFile):
        print("fileClassification: wav file not found!")
        return (-1, -1, -1)

    if model_type == 'knn':
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step,
         compute_beat] = load_model_knn(model_name)
    else:
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step,
         compute_beat] = load_model(model_name)

    [Fs, x] = audioBasicIO.readAudioFile(inputFile)        # read audio file and convert to mono
    x = audioBasicIO.stereo2mono(x)

    if isinstance(x, int):                                 # audio file IO problem
        return (-1, -1, -1)
    if x.shape[0] / float(Fs) <= mt_win:
        return (-1, -1, -1)

    # feature extraction:
    [mt_features, s, _] = aF.mtFeatureExtraction(x, Fs, mt_win * Fs, mt_step * Fs, round(Fs * st_win), round(Fs * st_step), feats)
    mt_features = mt_features.mean(axis=1)        # long term averaging of mid-term statistics
    if compute_beat:
        [beat, beatConf] = aF.beatExtraction(s, st_step)
        mt_features = numpy.append(mt_features, beat)
        mt_features = numpy.append(mt_features, beatConf)
    curFV = (mt_features - MEAN) / STD                # normalization

    [Result, P] = classifierWrapperHead(classifier, model_type, curFV)    # classification
    return Result, P, classNames


def fileRegression(inputFile, model_name, model_type, feats=["gfcc", "mfcc"]):
    # Load classifier:

    if not os.path.isfile(inputFile):
        print("fileClassification: wav file not found!")
        return (-1, -1, -1)

    regression_models = glob.glob(model_name + "_*")
    regression_models2 = []
    for r in regression_models:
        if r[-5::] != "MEANS":
            regression_models2.append(r)
    regression_models = regression_models2
    regression_names = []
    for r in regression_models:
        regression_names.append(r[r.rfind("_")+1::])

    # FEATURE EXTRACTION
    # LOAD ONLY THE FIRST MODEL (for mt_win, etc)
    if model_type == 'svm' or model_type == "svm_rbf" or model_type == 'randomforest':
        [_, _, _, mt_win, mt_step, st_win, st_step, compute_beat] = load_model(regression_models[0], True)

    [Fs, x] = audioBasicIO.readAudioFile(inputFile)        # read audio file and convert to mono
    x = audioBasicIO.stereo2mono(x)
    # feature extraction:
    [mt_features, s, _] = aF.mtFeatureExtraction(x, Fs, mt_win * Fs, mt_step * Fs, round(Fs * st_win), round(Fs * st_step), feats)
    mt_features = mt_features.mean(axis=1)        # long term averaging of mid-term statistics
    if compute_beat:
        [beat, beatConf] = aF.beatExtraction(s, st_step)
        mt_features = numpy.append(mt_features, beat)
        mt_features = numpy.append(mt_features, beatConf)

    # REGRESSION
    R = []
    for ir, r in enumerate(regression_models):
        if not os.path.isfile(r):
            print("fileClassification: input model_name not found!")
            return (-1, -1, -1)
        if model_type == 'svm' or model_type == "svm_rbf" \
                or model_type == 'randomforest':
            [model, MEAN, STD, mt_win, mt_step, st_win, st_step, compute_beat] = \
                load_model(r, True)
        curFV = (mt_features - MEAN) / STD                  # normalization
        R.append(regressionWrapper(model, model_type, curFV))    # classification
    return R, regression_names

def main(argv):
    return 0

if __name__ == '__main__':
    main(sys.argv)
