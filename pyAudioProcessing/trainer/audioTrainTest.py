"""
This script derives some of the functions from https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioTrainTest.py)

# TODO : add feature names and classifier names to the saved file.
"""
import sys
import numpy
import os
import pickle as cPickle
import signal

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from scipy.spatial import distance

import pyAudioProcessing.features.feature_computations as fc
from pyAudioProcessing.utils import convert_audio_to_mono, read_audio

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! - EXIT')
    os.system("stty -cbreak echo")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class kNN:
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k

    def classify(self, test_sample):
        n_classes = numpy.unique(self.Y).shape[0]
        y_dist = (distance.cdist(self.X,
                                 test_sample.reshape(1,
                                                     test_sample.shape[0]),
                                 'euclidean')).T
        i_sort = numpy.argsort(y_dist)
        P = numpy.zeros((n_classes,))
        for i in range(n_classes):
            P[i] = numpy.nonzero(self.Y[i_sort[0][0:self.k]] == i)[0].shape[0] / float(self.k)
        return (numpy.argmax(P), P)

def listOfFeatures2Matrix(features):
    '''
    This function takes a list of feature matrices as argument and returns a
    single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces
    '''

    X = numpy.array([])
    Y = numpy.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)

def load_model_knn(kNNModelName, is_regression=False):
    try:
        fo = open(kNNModelName, "rb")
    except Exception:
        print("didn't find file")
        return
    try:
        X = cPickle.load(fo)
        Y = cPickle.load(fo)
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        if not is_regression:
            classNames = cPickle.load(fo)
        K = cPickle.load(fo)
        mt_win = cPickle.load(fo)
        mt_step = cPickle.load(fo)
        st_win = cPickle.load(fo)
        st_step = cPickle.load(fo)
    except:
        fo.close()
    fo.close()

    X = numpy.array(X)
    Y = numpy.array(Y)
    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)

    classifier = kNN(X, Y, K)  # Note: a direct call to the kNN constructor is used here

    if is_regression:
        return(classifier, MEAN, STD, mt_win, mt_step, st_win, st_step)
    else:
        return(classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step)


def load_model(model_name, is_regression=False):
    '''
    This function loads an SVM model either for classification or training.
    ARGMUMENTS:
        - SVMmodel_name:     the path of the model to be loaded
        - is_regression:     a flag indigating whereas this model is regression or not
    '''
    try:
        fo = open(model_name + "MEANS", "rb")
    except Exception:
            print("Load SVM model: Didn't find file")
            return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        if not is_regression:
            classNames = cPickle.load(fo)
        mt_win = cPickle.load(fo)
        mt_step = cPickle.load(fo)
        st_win = cPickle.load(fo)
        st_step = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)

    with open(model_name, 'rb') as fid:
        SVM = cPickle.load(fid)

    if is_regression:
        return(SVM, MEAN, STD, mt_win, mt_step, st_win, st_step)
    else:
        return(SVM, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step)


def randSplitFeatures(features, per_train):
    '''
    This function splits a feature set for training and testing.

    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements
                            containt numpy matrices of features.
                            each matrix features[i] of class i is
                            [n_samples x numOfDimensions]
        - per_train:        percentage
    RETURNS:
        - featuresTrains:   a list of training data for each class
        - f_test:           a list of testing data for each class
    '''

    f_train = []
    f_test = []
    for i, f in enumerate(features):
        [n_samples, numOfDims] = f.shape
        randperm = numpy.random.permutation(range(n_samples))
        n_train = int(round(per_train * n_samples))
        f_train.append(f[randperm[0:n_train]])
        f_test.append(f[randperm[n_train::]])
    return f_train, f_test

def printConfusionMatrix(cm, class_names):
    '''
    This function prints a confusion matrix for a particular classification task.
    ARGUMENTS:
        cm:            a 2-D numpy array of the confusion matrix
                       (cm[i,j] is the number of times a sample from class i was classified in class j)
        class_names:    a list that contains the names of the classes
    '''

    if cm.shape[0] != len(class_names):
        print("printConfusionMatrix: Wrong argument sizes\n")
        return

    for c in class_names:
        if len(c) > 4:
            c = c[0:3]
        print("\t{0:s}".format(c), end="")
    print("")

    for i, c in enumerate(class_names):
        if len(c) > 4:
            c = c[0:3]
        print("{0:s}".format(c), end="")
        for j in range(len(class_names)):
            print("\t{0:.2f}".format(100.0 * cm[i][j] / numpy.sum(cm)), end="")
        print("")

def trainKNN(features, K):
    '''
    Train a kNN  classifier.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features.
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - K:                parameter K
    RETURNS:
        - kNN:              the trained kNN variable

    '''
    [Xt, Yt] = listOfFeatures2Matrix(features)
    knn = kNN(Xt, Yt, K)
    return knn

def trainSVM_RBF(features, Cparam):
    '''
    Train a multi-class probabilitistic SVM classifier.
    This function is simply a wrapper to the sklearn functionality for SVM training
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - Cparam:           SVM parameter C (cost of constraints violation)
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    svm = SVC(C = Cparam, kernel = 'rbf',  probability = True)
    svm.fit(X,Y)

    return svm


def trainRandomForest(features, n_estimators):
    '''
    Train a multi-class decision tree classifier.
    This function is simply a wrapper to the sklearn functionality for SVM training
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    rf = RandomForestClassifier(n_estimators = n_estimators)
    rf.fit(X,Y)

    return rf


def trainGradientBoosting(features, n_estimators):
    '''
    Train a gradient boosting classifier
    This function is simply a wrapper to the sklearn functionality for SVM training
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    '''
    [X, Y] = listOfFeatures2Matrix(features)
    rf = GradientBoostingClassifier(n_estimators = n_estimators)
    rf.fit(X,Y)
    return rf

def trainExtraTrees(features, n_estimators):
    '''
    Train a gradient boosting classifier
    Note:     This function is simply a wrapper to the sklearn functionality for extra tree classifiers
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    '''
    [X, Y] = listOfFeatures2Matrix(features)
    et = ExtraTreesClassifier(n_estimators = n_estimators)
    et.fit(X,Y)
    return et

def trainSVM(features, Cparam):
    '''
    Train a multi-class probabilitistic SVM classifier.
    This function is simply a wrapper to the sklearn functionality for SVM training
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [n_samples x numOfDimensions]
        - Cparam:           SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:              the trained SVM variable
    '''
    [X, Y] = listOfFeatures2Matrix(features)
    svm = SVC(C = Cparam, kernel = 'linear',  probability = True)
    svm.fit(X,Y)
    return svm

def normalizeFeatures(features):
    '''
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a numpy matrix)
    RETURNS:
        - features_norm:    list of NORMALIZED feature matrices
        - MEAN:        mean vector
        - STD:        std vector
    '''
    X = numpy.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1

    MEAN = numpy.mean(X, axis=0) + 0.00000000000001;
    STD = numpy.std(X, axis=0) + 0.00000000000001;

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - MEAN) / STD
        features_norm.append(ft)
    return (features_norm, MEAN, STD)

def classifierWrapper(classifier, classifier_type, test_sample):
    '''
    This function is used as a wrapper to pattern classification.
    ARGUMENTS:
        - classifier:        a classifier object of type SVC or kNN (defined in this library) or RandomForestClassifier or GradientBoostingClassifier  or ExtraTreesClassifier
        - classifier_type:    "svm" or "knn" or "randomforests" or "gradientboosting" or "extratrees"
        - test_sample:        a feature vector (numpy array)
    RETURNS:
        - R:            class ID
        - P:            probability estimate
    '''
    R = -1
    P = -1
    if classifier_type == "logisticregression":
        R = classifier.predict(test_sample.reshape(1,-1))[0]
        P = classifier.predict_proba(test_sample.reshape(1,-1))[0]
        return [R, P]
    elif classifier_type == "knn":
        [R, P] = classifier.classify(test_sample)
    elif classifier_type == "svm" or \
                    classifier_type == "randomforest" or \
                    classifier_type == "gradientboosting" or \
                    classifier_type == "extratrees" or \
                    classifier_type == "svm_rbf":
        R = classifier.predict(test_sample.reshape(1,-1))[0]
        P = classifier.predict_proba(test_sample.reshape(1,-1))[0]
    return [R, P]

def trainLogisticRegression(features, Cparam):
    '''
    Train a multi-class probabilitistic Logistic Regression classifier.
    This function is simply a wrapper to the sklearn functionality for
    logistic regression training
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    lr = LogisticRegression(C=Cparam, multi_class="ovr")
    lr.fit(X,Y)

    return lr

def evaluateclassifier(features, class_names, n_exp, classifier_name, Params, parameterMode, perTrain=0.90):
    '''
    Inputs
    features
        A list ([numOfClasses x 1]) whose elements contain numpy matrices of features.
        Each matrix features[i] of class i is [n_samples x numOfDimensions]
    class_names
        List of class names (strings)
    n_exp
        number of cross-validation experiments
    classifier_name: svm or knn or randomforest
    Params
        List of classifier parameters (for parameter tuning during cross-validation)
    parameterMode
        0: choose parameters that lead to maximum overall classification ACCURACY
        1: choose parameters that lead to maximum overall f1 MEASURE

    Returns
    bestParam
        The value of the input parameter that optimizes the selected performance measure
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
                    [res[ss], _] = classifierWrapper(classifier,
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
    list_of_dirs, mt_win, mt_step, st_win, st_step, feats, use_file_names=False, file_names={}
):
    """
    Extracts raw features specified by feats.
    Returns features, class names, file names and feature names.
    """
    [features, classNames, fileNames, featureNames] = fc.extract_features_from_audios(
        list_of_dirs,
        mt_win,
        mt_step,
        st_win,
        st_step,
        feats=feats,
        use_file_names=use_file_names,
        file_names=file_names
    )
    return features, classNames, fileNames, featureNames

def format_features(features):
    """
    Formats input list of features.
    """
    formatted_features = []
    for f in features:
        fTemp = []
        l = f.shape
        if len(l) == 1:
            shape = 1
        else:
            shape=l[0]
        for i in range(shape):
            if len(l) == 1:
                temp = f
            else:
                temp = f[i,:]
            if (not numpy.isnan(temp).any()) and (not numpy.isinf(temp).any()):
                fTemp.append(temp.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        formatted_features.append(numpy.array(fTemp))
    return formatted_features

def extract_features(
    list_of_dirs, mt_win, mt_step, st_win, st_step, feats, use_file_names=False, file_names={}
):
    """
    Extracts features and returns features, class names, file names
    and feature names.
    """
    features, classNames, fileNames, featureNames = extract_raw_features(
        list_of_dirs, mt_win, mt_step, st_win, st_step, feats, use_file_names=use_file_names, file_names=file_names
    )
    features = format_features(features)
    return features, classNames, fileNames, featureNames

def featureAndTrain(list_of_dirs, mt_win, mt_step, st_win, st_step,
                    classifier_type, model_name, perTrain=0.90, feats=["gfcc", "mfcc", "spectral", "chroma"],
                    use_file_names=False, file_names={}):
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
                                            feats=feats,
                                            use_file_names=use_file_names,
                                            file_names=file_names)

    if len(features) == 0:
        print("trainSVM_feature ERROR: No data found in any input folder!")
        return

    n_feats = features[0].shape[1]


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
    X = None
    Y = None
    if classifier_type == "knn":
        classifier = trainKNN(featuresNew, bestParam)
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
        fo.close()
    return classifier, classifier_type, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step, bestParam, X, Y

def fileClassification(inputFile, model_name, model_type, feats=["gfcc", "mfcc"]):
    # Load classifier:

    if not os.path.isfile(model_name):
        print("fileClassification: input model_name not found!")
        return (-1, -1, -1)

    if not os.path.isfile(inputFile):
        print("fileClassification: wav file not found!")
        return (-1, -1, -1)

    if model_type == 'knn':
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step] = load_model_knn(model_name)
    else:
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step] = load_model(model_name)

    [Fs, x] = read_audio(inputFile)        # read audio file and convert to mono
    x = convert_audio_to_mono(x)

    if isinstance(x, int):                                 # audio file IO problem
        return (-1, -1, -1)
    if x.shape[0] / float(Fs) <= mt_win:
        return (-1, -1, -1)

    # feature extraction:
    [mt_features, s, _] = fc.extract_agg_features(x, Fs, mt_win * Fs, mt_step * Fs, round(Fs * st_win), round(Fs * st_step), feats)
    mt_features = mt_features.mean(axis=1)        # long term averaging of mid-term statistics

    curFV = (mt_features - MEAN) / STD                # normalization

    [Result, P] = classifierWrapper(classifier, model_type, curFV)    # classification
    return Result, P, classNames
