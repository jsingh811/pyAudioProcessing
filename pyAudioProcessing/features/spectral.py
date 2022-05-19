#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
# Imports
################################################################################

import numpy as np

################################################################################
# Functions
################################################################################


def singal_energy(signal):
    return np.sum(signal ** 2)


def energy(signal):
    """
    Retuns the normalized energy of the signal
    """
    return singal_energy(signal) / np.float64(len(signal))


def zero_crossing_rate(signal):
    zeros = np.float64(np.sum(np.abs(np.diff(np.sign(signal)))) / 2)
    return zeros / np.float64(len(signal) - 1.0)


def roll_off(signal, thr):
    """
    Calculates the spectral roll-off of the input signal.
    The input domain is the freq domain representation of the audio.
    Spectral rolloff is the frequency location at which the
    spectral energy is equal to threshold * total signal energy
    """
    roll_off = 0.0
    energy = singal_energy(signal)
    energy_threshold = thr * energy
    cumsum = np.cumsum(signal ** 2) + 0.00000001
    opts = np.nonzero(cumsum > energy_threshold)[0]
    if len(opts) > 0:
        roll_off = np.float64(opts[0]) / len(signal)
    return roll_off


def flux(abs_fft, prev_abs_fft):
    """
    Calculates the spectral flux based on current and previous
    period signal's fft magnitude.
    """
    curr = abs_fft / np.sum(abs_fft + 0.00000001)
    prev = prev_abs_fft / np.sum(prev_abs_fft + 0.00000001)
    # sum of square distances
    flux = singal_energy((curr - prev))
    return flux


def entropy(signal, n=10):
    """
    Computes the spectral entropy
    """
    len_subframe = int(len(signal) / n)
    if len(signal) != len_subframe * n:
        signal = signal[0 : len_subframe * n]
    subframes = signal.reshape(len_subframe, n, order="F").copy()
    subframe_enery = np.sum(subframes ** 2, axis=0)
    # spectral sub-energies
    sub_energy = subframe_enery / (singal_energy(signal) + 0.00000001)
    # entropy
    entropy = -np.sum(sub_energy * np.log2(sub_energy + 0.00000001))
    return entropy


def get_spread(ind, centroid, norm_abs_fft, fs):
    spread = np.sqrt(
        np.sum(((ind - centroid) ** 2) * norm_abs_fft)
        / (np.sum(norm_abs_fft) + 0.00000001)
    )
    spread = spread / (fs / 2)
    return spread


def get_centroid(ind, norm_abs_fft):
    centroid = np.sum(ind * norm_abs_fft) / (np.sum(norm_abs_fft) + 0.00000001)
    return centroid


def centroid_and_spread(abs_fft, fs):
    """
    Calculates spectral centroid and spread of signal frame.
    abs_fft = signal magnitude in freq domain
    """
    ind = (np.arange(1, len(abs_fft) + 1)) * (fs / (2.0 * len(abs_fft)))

    Xt = abs_fft.copy()
    Xt_max = Xt.max()
    if Xt_max == 0:
        Xt = Xt / 0.00000001
    else:
        Xt = Xt / Xt_max
    centroid = get_centroid(ind, Xt)
    spread = get_spread(ind, centroid, Xt, fs)
    norm_centroid = centroid / (fs / 2)
    return norm_centroid, spread


def chroma(signal, fs):
    """
    A, A#, B, C, C#, D, D#, E, F, F#, G, G#
    Function derived from chroma implementation https://github.com/tyiannak/pyAudioAnalysis/blob/71c67d921aa0d059b57e95103d08aaf35c177efa/pyAudioAnalysis/ShortTermFeatures.py#L277
    """
    signal_len = len(signal)
    f = np.log2(
        np.array(
            [
                ((1 + inx) * fs) / (27.5 * (2 * signal_len))
                for inx in range(0, signal_len)
            ]
        )
    )
    num_chroma = np.round(f * 12).astype(int)
    num_f_chroma = np.zeros((num_chroma.shape[0],))
    for n in np.unique(num_chroma):
        inx = np.nonzero(num_chroma == n)
        num_f_chroma[inx] = inx[0].shape

    if num_chroma.max() < num_chroma.shape[0]:
        c = np.zeros((num_chroma.shape[0],))
        c[num_chroma] = signal ** 2
        c = c / num_f_chroma[num_chroma]
    else:
        i = np.nonzero(num_chroma > num_chroma.shape[0])[0][0]
        c = np.zeros((num_chroma.shape[0],))
        c[num_chroma[0 : i - 1]] = signal ** 2
        c = c / num_f_chroma
    final_matrix = np.zeros((12, 1))
    d = int(np.ceil(c.shape[0] / 12.0) * 12)
    c2 = np.zeros((d,))
    c2[0 : c.shape[0]] = c
    c2 = c2.reshape(int(c2.shape[0] / 12), 12)
    final_matrix = np.sum(c2, axis=0).reshape(1, -1).T

    if (signal ** 2).sum() == 0:
        return final_matrix / 0.00000001

    return final_matrix / ((signal ** 2).sum())
