# -*- coding: utf-8 -*-

"""
Created on Sat Aug 17 18:31:33 2019

@author: isancmen
"""

from preprocess.features import *
import numpy as np
from scipy.signal import resample

# Logging configuration
import logging

# Frequency analysis packages
from scipy.signal import periodogram

logger = logging.getLogger('MainLogger')


def extract_radio(L, n=None):
    """

    :param L:
    :param n:
    :return:
    """
    x = L['x'].values if n is None else resample(L['x'].values, n)
    y = L['y'].values if n is None else resample(L['y'].values, n)
    return radio(x, y)


def extract_residues(L, n=None, c=None):
    """

    :param L:
    :param n:
    :param c:
    :return:
    """
    x = L['x'].values if n is None else resample(L['x'].values, n)
    y = L['y'].values if n is None else resample(L['y'].values, n)
    rs_x = residues(x, c)
    rs_y = residues(y, c)
    return radio(rs_x, rs_y)


def extract_features_of(L):
    f, Pxx = periodogram(L, fs=1.0)
    return [
        # Time features
        # samp_ent(L), 
        # shannon_entrp(L), 
        mean_abs_val(L),
        np.var(L),
        root_mean_square(L)
        , log_detector(L)
        , wl(L)
        , np.nanstd(L)
        , diff_abs_std(L)
        , higuchi(L)
        , mfl(L)
        , myo(L)
        , iemg(L)
        , ssi(L)
        , zc(L)
        , ssc(L)
        , wamp(L)
        # # Frequency features
        , p_max(Pxx, L)
        , f_max(Pxx)
        , mp(Pxx)
        , tp(Pxx)
        , meanfreq(L)
        , medfreq(L)
        , std_psd(Pxx)
        , mmnt(Pxx, order=1)
        , mmnt(Pxx, order=2)
        , mmnt(Pxx, order=3)
        , kurt(Pxx)
        , skw(Pxx)
           # ,autocorr(L) # TODO
    ]