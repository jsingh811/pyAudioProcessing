### This script is derived https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioFeatureExtraction.py)
import time
import os
import glob
import numpy
import math
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import matplotlib.pyplot as plt

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.audioFeatureExtraction import (
    mfccInitFilterBanks,
    stChromaFeaturesInit,
    stChromaFeatures,
    stZCR,
    stEnergy,
    stEnergyEntropy,
    stSpectralCentroidAndSpread,
    stSpectralEntropy,
    stSpectralFlux,
    stSpectralRollOff,
    stMFCC,
    beatExtraction
)
from pyAudioProcessing.features import getGfcc


def stFeatureExtraction(signal, fs, win, step, feats):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        fs:           the sampling freq (in Hz)
        win:          the short-term window size (in samples)
        step:         the short-term window step (in samples)
        steps:        list of main features to compute ("mfcc" and/or "gfcc")
    RETURNS
        st_features:   a numpy array (n_feats x numOfShortTermWindows)
    """
    if "gfcc" in feats:
        ngfcc = 22
        gfcc = getGfcc.GFCCFeature(fs)
    else:
        ngfcc = 0

    if "mfcc" in feats:
        n_mfcc_feats = 13
    else: n_mfcc_feats = 0

    win = int(win)
    step = int(step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)                                # total number of samples
    cur_p = 0
    count_fr = 0
    nFFT = int(win / 2)

    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation

    n_harmonic_feats = 0

    feature_names = []
    if "spectral" in feats:
        n_time_spectral_feats = 8
        feature_names.append("zcr")
        feature_names.append("energy")
        feature_names.append("energy_entropy")
        feature_names += ["spectral_centroid", "spectral_spread"]
        feature_names.append("spectral_entropy")
        feature_names.append("spectral_flux")
        feature_names.append("spectral_rolloff")
    else:
        n_time_spectral_feats = 0
    if "mfcc" in feats:
        feature_names += ["mfcc_{0:d}".format(mfcc_i)
                      for mfcc_i in range(1, n_mfcc_feats+1)]
    if "gfcc" in feats:
        feature_names += ["gfcc_{0:d}".format(gfcc_i)
                      for gfcc_i in range(1, ngfcc+1)]
    if "chroma" in feats:
        nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, fs)
        n_chroma_feats = 13
        feature_names += ["chroma_{0:d}".format(chroma_i)
                          for chroma_i in range(1, n_chroma_feats)]
        feature_names.append("chroma_std")
    else:
        n_chroma_feats = 0
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + n_chroma_feats +ngfcc
    st_features = []
    while (cur_p + win - 1 < N):# for each short-term window until the end of signal
        count_fr += 1
        x = signal[cur_p:cur_p+win] # get current window
        cur_p = cur_p + step # update window position
        X = abs(fft(x)) # get fft magnitude
        X = X[0:nFFT] # normalize fft
        X = X / len(X)
        if count_fr == 1:
            X_prev = X.copy() # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((n_total_feats, 1))
        if "spectral" in feats:
            curFV[0] = stZCR(x) # zero crossing rate
            curFV[1] = stEnergy(x) # short-term energy
            curFV[2] = stEnergyEntropy(x) # short-term entropy of energy
            [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, fs)    # spectral centroid and spread
            curFV[5] = stSpectralEntropy(X) # spectral entropy
            curFV[6] = stSpectralFlux(X, X_prev) # spectral flux
            curFV[7] = stSpectralRollOff(X, 0.90, fs) # spectral rolloff
        if "mfcc" in feats:
            curFV[n_time_spectral_feats:n_time_spectral_feats+n_mfcc_feats, 0] = \
            stMFCC(X, fbank, n_mfcc_feats).copy()    # MFCCs
        if "gfcc" in feats:
            curFV[n_time_spectral_feats+n_mfcc_feats:n_time_spectral_feats+n_mfcc_feats+ngfcc, 0] = gfcc.get_gfcc(x)
        if "chroma" in feats:
            chromaNames, chromaF = stChromaFeatures(
                X, fs, nChroma, nFreqsPerChroma
            )
            curFV[n_time_spectral_feats + n_mfcc_feats + ngfcc:
                  n_time_spectral_feats + n_mfcc_feats + n_chroma_feats + ngfcc - 1] = \
                chromaF
            curFV[n_time_spectral_feats + n_mfcc_feats + n_chroma_feats + ngfcc - 1] = \
                chromaF.std()
        st_features.append(curFV)
        X_prev = X.copy()

    st_features = numpy.concatenate(st_features, 1)
    return st_features, feature_names


def mtFeatureExtraction(signal, fs, mt_win, mt_step, st_win, st_step, feats):
    """
    Mid-term feature extraction
    """

    mt_win_ratio = int(round(mt_win / st_step))
    mt_step_ratio = int(round(mt_step / st_step))

    mt_features = []

    st_features, f_names = stFeatureExtraction(signal, fs, st_win, st_step, feats)
    n_feats = len(st_features)
    n_stats = 2

    mt_features, mid_feature_names = [], []
    #for i in range(n_stats * n_feats + 1):
    for i in range(n_stats * n_feats):
        mt_features.append([])
        mid_feature_names.append("")

    for i in range(n_feats):        # for each of the short-term features:
        cur_p = 0
        N = len(st_features[i])
        mid_feature_names[i] = f_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = f_names[i] + "_" + "std"

        while (cur_p < N):
            N1 = cur_p
            N2 = cur_p + mt_win_ratio
            if N2 > N:
                N2 = N
            cur_st_feats = st_features[i][N1:N2]

            mt_features[i].append(numpy.mean(cur_st_feats))
            mt_features[i + n_feats].append(numpy.std(cur_st_feats))
            #mt_features[i+2*n_feats].append(numpy.std(cur_st_feats) / (numpy.mean(cur_st_feats)+0.00000010))
            cur_p += mt_step_ratio
    return numpy.array(mt_features), st_features, mid_feature_names


def dirWavFeatureExtraction(
    dirName,
    mt_win,
    mt_step,
    st_win,
    st_step,
    feats,
    compute_beat=False
):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder.

    The resulting feature vector is extracted by long-term averaging the mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mt_win, mt_step:    mid-term window and step (in seconds)
        - st_win, st_step:    short-term window and step (in seconds)
    """

    all_mt_feats = numpy.array([])
    process_times = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.mp3', '*.au', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(dirName, files)))

    wav_file_list = sorted(wav_file_list)
    wav_file_list2, mt_feature_names = [], []
    for i, wavFile in enumerate(wav_file_list):
        print("Analyzing file {0:d} of "
              "{1:d}: {2:s}".format(i+1,
                                    len(wav_file_list),
                                    wavFile))
        if os.stat(wavFile).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue
        [fs, x] = audioBasicIO.readAudioFile(wavFile)
        if isinstance(x, int):
            continue

        t1 = time.time()
        x = audioBasicIO.stereo2mono(x)
        if x.shape[0]<float(fs)/5:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        wav_file_list2.append(wavFile)
        if compute_beat:
            [mt_term_feats, st_features, mt_feature_names] = \
                mtFeatureExtraction(x, fs, round(mt_win * fs),
                                    round(mt_step * fs),
                                    round(fs * st_win), round(fs * st_step),
                                    feats)
            [beat, beat_conf] = beatExtraction(st_features, st_step)
        else:
            [mt_term_feats, _, mt_feature_names] = \
                mtFeatureExtraction(x, fs, round(mt_win * fs),
                                    round(mt_step * fs),
                                    round(fs * st_win), round(fs * st_step),
                                    feats)

        mt_term_feats = numpy.transpose(mt_term_feats)
        mt_term_feats = mt_term_feats.mean(axis=0)
        # long term averaging of mid-term statistics
        if (not numpy.isnan(mt_term_feats).any()) and \
                (not numpy.isinf(mt_term_feats).any()):
            if compute_beat:
                mt_term_feats = numpy.append(mt_term_feats, beat)
                mt_term_feats = numpy.append(mt_term_feats, beat_conf)
            if len(all_mt_feats) == 0:
                # append feature vector
                all_mt_feats = mt_term_feats
            else:
                all_mt_feats = numpy.vstack((all_mt_feats, mt_term_feats))
            t2 = time.time()
            duration = float(len(x)) / fs
            process_times.append((t2 - t1) / duration)
    if len(process_times) > 0:
        print("Feature extraction complexity ratio: "
              "{0:.1f} x realtime".format(
                  (1.0 / numpy.mean(numpy.array(process_times)))
            )
        )
    return (all_mt_feats, wav_file_list2, mt_feature_names)


def dirsWavFeatureExtraction(
    dirNames,
    mt_win,
    mt_step,
    st_win,
    st_step,
    feats,
    compute_beat=False
):
    '''
    Same as dirWavFeatureExtraction, but instead of a single dir it
    takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise','audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth','audioData/classSegmentsRec/shower'], 1, 1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in a separate path)
    '''

    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    feat_names = []
    for i, d in enumerate(dirNames):
        [f, fn, feature_names] = dirWavFeatureExtraction(d,
                                                         mt_win,
                                                         mt_step,
                                                         st_win,
                                                         st_step,
                                                         feats,
                                                         compute_beat=compute_beat)
        if f.shape[0] > 0:
            # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            feat_names.append(feature_names)
            if d[-1] == os.sep:
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    
    return features, classNames, fileNames, feat_names
