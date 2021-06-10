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

from pyAudioProcessing.features import filters

################################################################################
# Globals
################################################################################

CEP_COEF_START = 1
CEP_COEF_END = 23

################################################################################
# Classes
################################################################################

class GFCCFeature(object):
    """
    With sampling frequency as input, creates ERB Filter and gets
    Gammatone Frequency Cepstral Coefficients (GFCC) for an input
    audio signal window.
    """
    def __init__(
        self,
        fs,
        cc_start=CEP_COEF_START,
        cc_end=CEP_COEF_END
    ):
        self.fs = fs
        self.erb_filter = self.erb_filter()
        self.ccST = cc_start
        self.ccEND = cc_end

    def dct_matrix(self, n):
        """
        Return the DCT-II matrix of order n as a numpy array.
        """
        x, y = numpy.meshgrid(range(n), range(n))
        D = math.sqrt(2.0 / n) * numpy.cos(
            math.pi * (2*x+1) * y / (2*n)
        )
        D[0] /= math.sqrt(2)
        return D

    def erb_filter(self):
        """
        For the input sampling frequency, get the ERB filters.
        """
        erb_filter_result = filters.make_erb_filters(
            self.fs,
            filters.centre_freqs(self.fs, 64, 50)
        )
        
        return erb_filter_result

    def mean_var_norm(self, x, std=True):
        """
        Returns mean variance normalization.
        """
        norm = x - numpy.mean(x, axis=0)
        
        if std is True:
            norm = norm / numpy.std(norm)
            
        return norm

    def get_gfcc(
        self,
        signal,
        norm=False
    ):
        """
        Get GFCC feature.
        """
        erb_filterbank = filters.erb_filterbank(
            numpy.array(signal),
            self.erb_filter
        )
        inData = erb_filterbank[10:,:]
        inData = numpy.absolute(inData)
        inData = numpy.power(inData, 1/3)
        [chnNum, frmNum] = numpy.array(inData).shape
        mtx = self.dct_matrix(chnNum)
        outData = numpy.matmul(mtx, inData)
        outData = outData[self.ccST:self.ccEND, :]
        gfcc_feat = numpy.array(
            [numpy.mean(data_list) for data_list in outData]
        ).copy()
        
        if norm is True:
            gfcc_feat = self.mean_var_norm(gfcc_feat)
        
        return gfcc_feat
