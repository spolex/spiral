# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:31:33 2019

@author: isancmen
"""

from interfaces.arq_loader import load_arquimedes_dataset
from interfaces.reader_and_writer import save
from preprocess.features import *
import numpy as np
from scipy.signal import resample

# Logging configuration
import logging

# Frequency analysis packages
from scipy.signal import periodogram

logger = logging.getLogger('MainLogger')


def extract_radio(L, n=None):
    x = L['x'].values if n is None else resample(L['x'].values, n)
    y = L['y'].values if n is None else resample(L['y'].values, n)
    rho, phi = cart2pol(x, y)
    return radio(rho, phi)


def extract_residuos(L, n=None):
    x = L['x'].values if n is None else resample(L['x'].values, n)
    y = L['y'].values if n is None else resample(L['y'].values, n)
    rs_x = residuos(x)
    rs_y = residuos(y)
    return radio(rs_x, rs_y)


def extract_features_of(L, ts):
    f, Pxx = periodogram(L, fs=1.0)
    L = L.values
    ts = ts.values
    return [
        # Time features
        samp_ent(L)
        , mean_abs_val(L)
        , np.var(L)
        , root_mean_square(L)
        , log_detector(L)
        , wl(L)
        , np.nanstd(L)
        , diff_abs_std(L)
        , higuchi(L)
        , mfl(L)
        , myo(L, ts)
        , iemg(L)
        , ssi(L)
        , zc(L)
        , ssc(L)
        , wamp(L)
        # ,yuel_walker(L) #TODO
        # Frequency features
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
        #    ,autocorr(L) # TODO
    ]


def extract_features(filenames_file, root_ct, root_et, h5file, samples=4096):
    logger.debug("Starting feature extraction from archimedean spirals")
    #   loadin files
    logger.debug("Loading control files")
    ct = load_arquimedes_dataset(filenames_file, root_ct)
    logger.debug("Loading CT files %d", len(ct))
    ct_ts = list(map(lambda df: df['timestamp'], ct))

    ##Controls##

    #    residual radio calculation
    rd_ct = np.array(list(map(lambda c: extract_residuos(c, samples), ct)))
    logger.debug("CT's residual radio calculation %d", len(rd_ct))

    #    rd feature extraction
    rd_ct_fe = np.array(list(map(extract_features_of, rd_ct, ct_ts)))
    logger.debug("CT's residual feature extraction %i", len(rd_ct_fe[0]))
    logger.debug("Saving CT's residual feature extraction in " + h5file)
    save(h5file, 'rd_ct_fe', rd_ct_fe)

    #   polar radio calculation
    r_ct = np.array(list(map(lambda c: extract_radio(c, samples), ct)))
    logger.debug("CT's radio calculation %d", len(r_ct))

    #   radio feature extraction
    r_ct_fe = list(map(extract_features_of, r_ct, ct_ts))
    logger.debug("CT's radio feature extraction %i", len(r_ct_fe[0]))
    logger.debug("Saving CT's radio feature extraction in " + h5file)
    save(h5file, 'r_ct_fe', r_ct_fe)

    ##ET##
    #   loadin files
    logger.debug("Loading ET files")
    et = load_arquimedes_dataset(filenames_file, root_et)
    et_ts = list(map(lambda df: df['timestamp'], et))

    #    residual radio calculation
    rd_et = np.array(list(map(lambda c: extract_residuos(c, samples), et)))
    logger.debug("ET's residual radio calculation %d", len(rd_et))

    #    rd feature extraction
    rd_et_fe = np.array(list(map(extract_features_of, rd_et, et_ts)))
    logger.debug("TT's residual feature extraction %i", len(rd_et_fe))
    logger.debug("Saving TT's residual feature extraction in " + h5file)
    save(h5file, 'rd_et_fe', rd_et_fe)

    #   radio calculation
    r_et = np.array(list(map(lambda c: extract_radio(c, samples), et)))
    logger.debug("ET's radio calculation %d", len(r_et))

    #   radio feature extraction
    r_et_fe = list(map(extract_features_of, r_et, et_ts))
    logger.debug("ET's radio feature extraction %i", len(r_et_fe[0]))
    logger.debug("Saving ET's radio feature extraction in " + h5file)
    save(h5file, 'r_et_fe', r_et_fe)
