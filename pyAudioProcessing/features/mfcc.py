#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
# Imports
################################################################################

import numpy as np
from scipy.fftpack.realtransforms import dct

################################################################################
# Globals
################################################################################

LEN = 13

################################################################################
# Functions
################################################################################


def triangular_filter_bank(
    fs, nfft, lowfreq=133.33, linc=200/3, logsc=1.0711703, lin_filt=LEN, log_filt=27
):
    """
    Triangular filterbank for MFCC computation
    lin_filt = No. of linear filters
    log_filt = No. of log filters
    """
    num_filts = lin_filt + log_filt

    # start, mid and end points of the filters is spectral domain
    freqs = np.zeros(num_filts + 2)
    freqs[0:lin_filt] = lowfreq + np.arange(lin_filt) * linc
    freqs[lin_filt:] = freqs[lin_filt - 1] * logsc ** np.arange(1, log_filt + 3)
    denom = freqs[2:] - freqs[0:-2]
    heights = 2.0 / denom

    # filterbank coeff (fft bins Hz)
    fbank = np.zeros((num_filts, nfft))
    nfreqs = np.arange(nfft) / (1.0 * nfft) * fs

    for i in range(num_filts):

        high_freqs = freqs[i + 2]

        lid = np.arange(
            np.floor(freqs[i] * nfft / fs) + 1,
            np.floor(freqs[i + 1] * nfft / fs) + 1,
            dtype=int,
        )
        lslope = heights[i] / (freqs[i + 1] - freqs[i])
        rid = np.arange(
            np.floor(freqs[i + 1] * nfft / fs) + 1,
            np.floor(high_freqs * nfft / fs) + 1,
            dtype=int,
        )
        rslope = heights[i] / (freqs[i + 2] - freqs[i + 1])
        fbank[i][lid] = lslope * (nfreqs[lid] - freqs[i])
        fbank[i][rid] = rslope * (freqs[i + 2] - nfreqs[rid])

    return fbank, freqs


def compute_mfcc(fft_magnitude, fbank, num_mfcc_feats):
    """
    Returns the MFCCs of a signal frame.
    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence).
    """
    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + 0.00000001)
    ceps = dct(mspec, type=2, norm="ortho", axis=-1)[:num_mfcc_feats]

    return ceps


def get_mfcc(fs, nfft, n_mfcc_feats, signal):
    """
    Returns mfcc features for a signal.
    """
    [fbank, freqs] = triangular_filter_bank(fs, nfft)
    feature = compute_mfcc(signal, fbank, n_mfcc_feats)
    return feature
