# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:31:33 2019

@author: isancmen
"""

from loaders.biodarw_loader import load_arquimedes_dataset
from loaders.reader_and_writer import save, load
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


def extract_residuos(L, n=None, c=None):
    x = L['x'].values if n is None else resample(L['x'].values, n)
    y = L['y'].values if n is None else resample(L['y'].values, n)
    rs_x = residuos(x, c)
    rs_y = residuos(y, c)
    return radio(rs_x, rs_y)


def extract_features_of(L):
    f, Pxx = periodogram(L, fs=1.0)
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
        , myo(L)
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


def extract_rr(filenames, root_ct, root_et, h5file, coeff=None, samples=4096):
    """

    :param filenames:
    :param root_ct:
    :param root_et:
    :param h5file:
    :param samples:
    :return:
    """

    logger.debug("Starting radius and DCT calculations extraction from archimedean spirals")

    start_time = time.time()
    logger.debug("Loading Controls files")
    ct = load_arquimedes_dataset(filenames, root_ct)
    logger.debug("Loading ET files")
    et = load_arquimedes_dataset(filenames, root_et)

    logger.debug("Polar radius calculation")

    r_ct = np.array(list(map(lambda c: extract_radio(c, samples), ct)))
    logger.debug("CT's polar radius calculation %d", len(r_ct))
    save(h5file, 'r_ct', r_ct)

    r_et = np.array(list(map(lambda c: extract_radio(c, samples), et)))
    logger.debug("ET's polar radius calculation %d", len(r_et))
    save(h5file, 'r_et', r_et)

    logger.debug("Residual radius calculation")

    rd_ct = np.array(list(map(lambda c: extract_residuos(c, samples, coeff), ct)))
    logger.debug("CT's residual radius calculation %d", len(rd_ct))
    save(h5file, 'rd_ct', rd_ct)

    rd_et = np.array(list(map(lambda c: extract_residuos(c, samples, coeff), et)))
    logger.debug("ET's residual radius calculation %d", len(rd_et))
    save(h5file, 'rd_et', rd_et)

    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate polar and residual radius is %s", elapsed_time)


def extract_features(h5file):

    logger.debug("Starting feature extraction from archimedean spirals")
    logger.debug("Extracting timeseries")

    r_ct = load(h5file, 'r_ct', 'r')
    r_ct_fe = list(map(extract_features_of, r_ct))
    logger.debug("CT's radius feature extraction %i", len(r_ct_fe[0]))
    logger.debug("Saving CT's radius feature extraction in " + h5file)
    save(h5file, 'r_ct_fe', r_ct_fe)

    r_et = load(h5file, 'r_et', 'r')
    r_et_fe = list(map(extract_features_of, r_et))
    logger.debug("ET's radius feature extraction %i", len(r_et_fe[0]))
    logger.debug("Saving ET's radius feature extraction in " + h5file)
    save(h5file, 'r_et_fe', r_et_fe)

    rd_ct = load(h5file, 'rd_ct', 'r')
    rd_ct_fe = np.array(list(map(extract_features_of, rd_ct)))
    logger.debug("CT's residual feature extraction %i", len(rd_ct_fe[0]))
    logger.debug("Saving CT's residual feature extraction in " + h5file)
    save(h5file, 'rd_ct_fe', rd_ct_fe)

    rd_et = load(h5file, 'rd_et', 'r')
    rd_et_fe = np.array(list(map(extract_features_of, rd_et)))
    logger.debug("TT's residual feature extraction %i", len(rd_et_fe))
    logger.debug("Saving TT's residual feature extraction in " + h5file)
    save(h5file, 'rd_et_fe', rd_et_fe)