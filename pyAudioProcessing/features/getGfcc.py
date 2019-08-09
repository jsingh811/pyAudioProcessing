#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:08:51 2019

@author: jsingh
"""
################################################################################
# Imports
################################################################################
import math
import numpy

from gammatone import filters

################################################################################
# Classes
################################################################################

class GFCCFeature(object):
    """
    With sampling frequency as input, creates ERB Filter and gets
    Gammatone Frequency Cepstral Coefficients (GFCC) for an input audio signal window.
    """
    def __init__(self, fs):
        self.fs = fs
        self.erb_filter = self.erb_filter()

    def dct_matrix(self, n):
        """
        Return the DCT-II matrix of order n as a numpy array.
        """
        x, y = numpy.meshgrid(range(n), range(n))
        D = math.sqrt(2.0 / n) * numpy.cos(math.pi * (2*x+1) * y / (2*n))
        D[0] /= math.sqrt(2)
        return D

    def erb_filter(self):
        """
        For the input sampling frequency, get the ERB filters.
        """
        return filters.make_erb_filters(self.fs, filters.centre_freqs(self.fs, 64, 50))

    def get_gfcc(self, signal, ccST=1, ccEND=23):
        """
        Get GFCC feature.
        """
        erb_filterbank = filters.erb_filterbank(numpy.array(signal), self.erb_filter)
        inData = erb_filterbank[10:,:]
        [chnNum, frmNum] = numpy.array(inData).shape
        mtx = self.dct_matrix(chnNum)
        outData = numpy.matmul(mtx, inData)
        outData = outData[ccST:ccEND, :]
        gfcc_feat = numpy.array(
            [numpy.mean(data_list) for data_list in outData]
        ).copy()
        return gfcc_feat
