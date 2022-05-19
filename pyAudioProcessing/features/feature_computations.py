#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:54:34 2022

@author: jsingh
"""
################################################################################
# Imports
################################################################################

import os
import glob
import numpy as np
from scipy.fftpack import fft

from pyAudioProcessing.features import getGfcc
from pyAudioProcessing.features import mfcc
from pyAudioProcessing.features import spectral
from pyAudioProcessing.utils import convert_audio_to_mono, read_audio

################################################################################
# Classes
################################################################################


def extract_st_features(signal, fs, win, step, feats):
    """
    Windows the signal and get features for each window in time and freq domain.
    """
    # signal normalization
    signal = np.double(signal) / (2.0 ** 15)
    sig_max = (np.abs(signal)).max()
    signal = (signal - signal.mean()) / (sig_max + 0.0000000001)

    sig_len = len(signal)
    st_features = []
    feature_names = []
    for pos in range(0, sig_len + 1 - win, step):
        window_signal = signal[pos : pos + win]
        freq_domain_signal = (abs(fft(window_signal))[0 : int(win / 2)]) / int(win / 2)
        win_features = np.array([])
        if "spectral" in feats:
            if len(win_features) == 0:
                win_features = np.array([[spectral.zero_crossing_rate(window_signal)]])
            else:
                win_features = np.append(
                    win_features, [[spectral.zero_crossing_rate(window_signal)]], axis=0
                )
            win_features = np.append(
                win_features, [[spectral.energy(window_signal)]], axis=0
            )
            win_features = np.append(
                win_features, [[spectral.entropy(window_signal)]], axis=0
            )
            ce, sp = spectral.centroid_and_spread(freq_domain_signal, fs)
            win_features = np.append(win_features, [[ce], [sp]], axis=0)
            win_features = np.append(
                win_features, [[spectral.entropy(freq_domain_signal)]], axis=0
            )
            if pos == 0:
                freq_domain_signal_prev = freq_domain_signal.copy()
            win_features = np.append(
                win_features,
                [[spectral.flux(freq_domain_signal, freq_domain_signal_prev)]],
                axis=0,
            )
            win_features = np.append(
                win_features, [[spectral.roll_off(freq_domain_signal, fs)]], axis=0
            )
            if pos == 0:
                feature_names += [
                    "zcr",
                    "energy",
                    "energy_entropy",
                    "spectral_centroid",
                    "spectral_spread",
                    "spectral_entropy",
                    "spectral_flux",
                    "spectral_rolloff",
                ]
        if "mfcc" in feats:
            mfcc_feat = np.array(
                [
                    [i]
                    for i in mfcc.get_mfcc(
                        fs, int(win / 2), 13, freq_domain_signal
                    ).copy()
                ]
            )
            if len(win_features) == 0:
                win_features = mfcc_feat
            else:
                win_features = np.append(
                    win_features,
                    np.array(
                        [
                            [i]
                            for i in mfcc.get_mfcc(
                                fs, int(win / 2), 13, freq_domain_signal
                            ).copy()
                        ]
                    ),
                    axis=0,
                )
            if pos == 0:
                feature_names += [
                    "mfcc_{0:d}".format(mfcc_i) for mfcc_i in range(1, 13 + 1)
                ]
        if "gfcc" in feats:
            gfcc_feat = np.array(
                [[i] for i in getGfcc.GFCCFeature(fs).get_gfcc(window_signal)]
            )
            if len(win_features) == 0:
                win_features = gfcc_feat
            else:
                win_features = np.append(win_features, gfcc_feat, axis=0)
            if pos == 0:
                feature_names += [
                    "gfcc_{0:d}".format(gfcc_i)
                    for gfcc_i in range(1, len(gfcc_feat) + 1)
                ]
        if "chroma" in feats:
            chroma = spectral.chroma(freq_domain_signal, fs)
            if len(win_features) == 0:
                win_features = np.array(chroma)
            else:
                win_features = np.append(win_features, chroma, axis=0)
            win_features = np.append(win_features, [[chroma.std()]], axis=0)
            if pos == 0:
                feature_names += [
                    "chroma_{0:d}".format(chroma_i)
                    for chroma_i in range(1, len(chroma) + 1)
                ]
                feature_names.append("chroma_std")
        st_features.append(win_features)
        freq_domain_signal_prev = freq_domain_signal.copy()

    st_features = np.array(st_features)
    shape = st_features.shape
    st_features = st_features.reshape(shape[0], shape[1])
    st_features = st_features.T
    return st_features, feature_names


def extract_agg_features(signal, fs, mt_win, mt_step, st_win, st_step, feats):
    """
    Returns mean and standard deviation of every feature.
    """
    agg_features = []

    st_features, feature_names = extract_st_features(signal, fs, st_win, st_step, feats)
    num_feats = len(st_features)

    for i in range(2 * num_feats):
        agg_features.append([])

    feat_names = [
        "_".join([feature_names[feat_num], "mean"]) for feat_num in range(num_feats)
    ] + ["_".join([feature_names[feat_num], "std"]) for feat_num in range(num_feats)]

    for i in range(num_feats):
        feat_len = len(st_features[i])
        inx = 0
        while inx < feat_len:
            pos1 = inx
            pos2 = inx + int(round(mt_win / st_step))
            pos2 = min(pos2, feat_len)  # capping max
            st = st_features[i][pos1:pos2]
            # append mean and std
            agg_features[i].append(np.mean(st))
            agg_features[i + num_feats].append(np.std(st))
            # update index to next step
            inx += int(round(mt_step / st_step))
    return np.array(agg_features), st_features, feat_names


def extract_features_from_audio_locations(
    wav_file_list, mt_win, mt_step, st_win, st_step, feats
):
    """
    Extracts averaged / mid-term features from audio paths in wav_file_list.
    """
    agged_features = np.array([])
    wav_file_list2, mt_feature_names = [], []
    for i, file in enumerate(wav_file_list):
        print(f"Computing features for file {i+1} of {len(wav_file_list)} : {file}")
        if os.stat(file).st_size == 0:
            print("....Audio file is empty. Skipping...")
            continue
        [fs, x] = read_audio(file)
        if isinstance(x, int):
            continue
        x = convert_audio_to_mono(x)
        if x.shape[0] < fs / 5:
            print("....Audio file is too small for analysis. Skipping...")
            continue
        wav_file_list2.append(file)
        [mt_term_feats, _, mt_feature_names] = extract_agg_features(
            x,
            fs,
            round(mt_win * fs),
            round(mt_step * fs),
            round(fs * st_win),
            round(fs * st_step),
            feats,
        )

        mt_term_feats = np.transpose(mt_term_feats)
        mt_term_feats = mt_term_feats.mean(axis=0)
        # long term averaging of the mid-term feature statistics
        if (not np.isnan(mt_term_feats).any()) and (not np.isinf(mt_term_feats).any()):
            if len(agged_features) == 0:
                agged_features = mt_term_feats
            else:
                # append
                agged_features = np.vstack((agged_features, mt_term_feats))
    return (agged_features, wav_file_list2, mt_feature_names)


def extract_features_from_audios(
    dirpath,
    mt_win,
    mt_step,
    st_win,
    st_step,
    feats=["mfcc", "gfcc"],
    use_file_names=False,
    file_names={},
):
    """
    Extract audio features from either sub-folders in parent folder specified by
    dirpath, or by using file locations specified by input file_names. file_names
    is only used if use_file_names is True.
    """
    features = []
    labels = []
    filepaths = []
    feature_labels = []
    wav_file_list = []
    if use_file_names:
        for d in file_names:
            wav_file_list = []
            for file in file_names[d]:
                if file.endswith(".wav"):
                    wav_file_list.append(file)
            wav_file_list = sorted(wav_file_list)
            [f, fn, feature_names] = extract_features_from_audio_locations(
                wav_file_list, mt_win, mt_step, st_win, st_step, feats
            )
            if f.shape[0] > 0:
                # at least 1 valid audio file has been found
                features.append(f)
                filepaths.append(fn)
                feature_labels.append(feature_names)
                if d[-1] == os.sep:
                    labels.append(d.split(os.sep)[-2])
                else:
                    labels.append(d.split(os.sep)[-1])
    else:
        for i, d in enumerate(dirpath):
            wav_file_list = []
            for ext in ["*.wav"]:  #'*.aif',  '*.aiff', '*.mp3', '*.au', '*.ogg'
                wav_file_list.extend(glob.glob(os.path.join(d, ext)))
            wav_file_list = sorted(wav_file_list)
            [f, fn, feature_names] = extract_features_from_audio_locations(
                wav_file_list, mt_win, mt_step, st_win, st_step, feats
            )
            if f.shape[0] > 0:
                # at least 1 valid audio file has been found
                features.append(f)
                filepaths.append(fn)
                feature_labels.append(feature_names)
                if d[-1] == os.sep:
                    labels.append(d.split(os.sep)[-2])
                else:
                    labels.append(d.split(os.sep)[-1])
    return features, labels, filepaths, feature_labels
