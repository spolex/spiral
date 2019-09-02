# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:41:36 2019

@author: isancmen
higuchi_fd

"""
from sampen import sampen2
import hfda
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import moment, kurtosis, skew
from math import log, pi
from functools import partial
import time


import logging

logger = logging.getLogger('MainLogger')


def idct_vec(n, Xk, l, N):
    """

    :param n:
    :param Xk:
    :param l:
    :param N:
    :return:
    """
    start_time = time.time()
    k = np.arange(l)
    c = np.zeros(k.shape)
    c[0] = (1 / N) ** (1 / 2)
    c[1:] = (2 / N) ** (1 / 2)
    f = (pi / N) * (n + (1 / 2)) * k
    logger.debug("Length of array function is %s", len(f))
    cs = np.cos(f)
    logger.debug("Length of array cosin is %s", len(cs))
    xn = c * Xk * cs
    logger.debug("Length of xn is %s", len(xn))
    rdo = xn.sum()
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate idct_vec  is %s", elapsed_time)
    return rdo


def idct(Xk, l=None):
    """

    :param Xk:
    :param l:
    :return:
    """
    start_time = time.time()
    N = len(Xk)
    if l != None:
        assert N > l, "l can not be bigger than len of Xk"
    l = l if l != None else N

    Xk = Xk[:l]
    n = np.arange(N)
    X = np.array(list(map(partial(idct_vec, Xk=Xk, l=l, N=N), n)))
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate idct  is %s", elapsed_time)
    return X


def dct_vect(k, N, X):
    """

    :param k:
    :param N:
    :param X:
    :return:
    """
    start_time = time.time()
    n = np.arange(N)
    c = np.zeros(n.shape)
    c[0] = (1 / N) ** (1 / 2)
    c[1:] = (2 / N) ** (1 / 2)
    f = (pi / N) * (n + (1 / 2)) * k
    logger.debug("Length of array function is %s", len(f))
    cs = np.cos(f)
    logger.debug("Length of array cosin is %s", len(cs))
    xk = c * X * cs
    logger.debug("Length of xk is %s", len(xk))
    rdo = xk.sum()
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate dct_vect  is %s", elapsed_time)
    return rdo


def dct(X):
    """

    :param X:
    :return:
    """
    start_time = time.time()
    N = len(X)
    # TODO validate vectorial implementation
    k = np.arange(N)
    Xk = np.array(list(map(partial(dct_vect, N=N, X=X), k)))
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate dct  is %s", elapsed_time)
    # sequencial implementation:
    #    Xk = np.zeros(X.shape)
    #    for k in range(N):
    #        n = np.arange(N)
    #        c = (1/N)**(1/2) if k == 0 else (2/N)**(1/2)
    #        f = (pi/N)*(n+(1/2))*k
    #        logger.debug("Length of array function is %s",len(f))
    #        cs = np.cos(f)
    #        logger.debug("Length of array cosin is %s",len(cs))
    #        xk = c*X*cs
    #        logger.debug("Length of xk is %s",len(xk))
    #        Xk[k]=xk.sum()
    return Xk


def radio(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    start_time = time.time()
    r = (x ** 2 + y ** 2) ** (1 / 2)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate radio(x,y) is %s", elapsed_time)
    return r


def residuos(x, l=17):
    """
    # TODO https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
    :param x:
    :param l:
    :return:
    """
    start_time = time.time()
    idct_x = idct(dct(x), l=l)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate residuos(x, l=17) is %s", elapsed_time)
    return x - idct_x


def cart2pol(x, y):
    """
    From cartesian coordinates to polar coordinates
    :param x: x coordinate
    :param y: y coordinate
    :return: polar coordinates (rho, phi)
    """
    start_time = time.time()
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to transformate from cartesian coordinates to polar coordinates is %s", elapsed_time)
    return (rho, phi)


def pol2cart(rho, phi):
    """
    From polar coordinates to cartesian coordinates
    :param rho: rho coordinate
    :param phi: phi coordinate
    :return: cartesian coordinates (x,y)
    """
    start_time = time.time()
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to transformate from polar coordinates to cartesian coordinates is %s", elapsed_time)
    return (x, y)


# sample entropy
def samp_ent(r):
    """
    https://sampen.readthedocs.io/en/stable/#documentation
    :param r: time series data
    :return: sample entropy
    """
    start_time = time.time()
    se = sampen2(r, mm=3, r=2)
    elapsed_time = time.time() - start_time
    logger.info("Sample Entropy")
    logger.debug("Elapsed time to calculate sample entropy is %s", elapsed_time)
    return se[len(se) - 1][1]


# mean absolute value
def mean_abs_val(L):
    """
    :param L:
    :return:
    """
    logger.info("mean absolute value")
    start_time = time.time()
    N = len(L)
    mav = (1 / N) * L.sum()
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate mean absolute value is %s", elapsed_time)
    return mav


### Difference Absolute standard deviation (AAC)
def diff_abs_std(L):
    """

    :param L:
    :return:
    """
    start_time = time.time()
    N = len(L)
    logger.debug("Timeseries' max is %s", max(L))
    den = 1 / (N - 1)
    square_diff = (np.diff(L)) ** 2
    logger.debug("Square differences max %s", max(square_diff))
    avg_diff = den * sum(square_diff)
    logger.debug("Avg differences is %s", avg_diff)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate Difference Absolute standard deviation (AAC) is %s", elapsed_time)
    aac = avg_diff ** (1 / 2)
    return aac


## log detector
def log_detector(L):
    """

    :param L:
    :return:
    """
    N = len(L)
    f = (1 / N) * np.log(np.abs(L)).sum()
    return np.nanstd(L) ** (f)


### Waveform length (WL)
def wl(L):
    """

    :param L:
    :return:
    """
    return sum(abs(np.diff(L)))


### Root mean square
def root_mean_square(L):
    """

    :param L:
    :return:
    """
    N = len(L)
    return ((1 / N) * L ** 2).sum()


### higuchi fractal dimension
def higuchi(L):
    """

    :param L:
    :return:
    """
    if logger.getEffectiveLevel() == logging.DEBUG:
        start_time = time.time()
        h = hfda.measure(L, 5)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time to calculate higuchi fractal dimension is %s", elapsed_time)
        return h
    else:
        logger.info("higuchi Fractal Dimension")
        return hfda.measure(L, 5)


### Wilson amplitude (WAMP)
def wamp(L):
    """

    :param L:
    :return:
    """
    logger.info("Wilson amplitude (WAMP)")
    epsilon = L.mean()
    a = L[:-1]
    b = L[1:]
    diff = abs(a - b)
    amp = diff > epsilon
    return amp.sum()


### Maximum fractal length (MFL)
def mfl(L):
    """

    :param L:
    :return:
    """
    logger.info("Maximum fractal length (MFL)")
    return log(wl(L))


### Myopulse percentage rate (MYO)
def myo(L, ts):
    """
    Percentage of time where the signal is bigger than two times the mean
    :param L:
    :param ts:
    :return:
    """
    assert len(ts) > 0, "Myopulse percentage rate needs timestamp"
    logger.info("Myopulse percentage rate (MYO)")
    N = len(L)
    dos_epsilon = 2 * L.mean()
    biggers = L[L > dos_epsilon]
    logger.debug("Myopulse times rate (MYO) %s", len(biggers))
    myo = len(biggers) / N
    logger.debug("Myopulse percentage rate (MYO) %s", myo)
    return myo


### Integrated EMG (IEMG)
def iemg(L):
    """

    :param L:
    :return:
    """
    logger.info("Integrated EMG (IEMG)")
    return abs(L).sum()


### Simple square EMG (SSI)
def ssi(L):
    """

    :param L:
    :return:
    """
    logger.info("Simple square EMG (SSI)")
    return (L ** 2).sum()


### Zero crossing (ZC)
def zc(L, epsilon=None):
    """

    :param L:
    :param epsilon:
    :return:
    """
    """
    Description: The number of times in which the signal crosses its mean
    """
    logger.info("Zero crossing (ZC)")
    if (epsilon == None):
        epsilon = L.mean()
    return len(L[L > epsilon])


def masks(L):
    d = np.diff(L)
    dd = np.diff(d)
    # Mask of locations where graph goes to vertical or horizontal, depending on vec
    to_mask = ((d[:-1] != 0) & (d[:-1] == -dd))
    # Mask of locations where graph comes from vertical or horizontal, depending on vec
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask


### Slope sign change (SSC)
def ssc(L):
    logger.info("Slope sign change (SSC)")
    to_vert_mask, from_vert_mask = masks(L)
    return sum(to_vert_mask) + sum(from_vert_mask)


### Main peak amplitude (Pmax)
def p_max(psd, L):
    """
    Maximum peak of frequency
    """
    logger.info("Main peak amplitude (Pmax)")
    peaks, _ = find_peaks(psd)
    p_max = max(L[peaks])
    return p_max


### Main peak frequency (Fmax)
def f_max(PSD):
    """
    Frequency of the max peak
    """
    logger.info("Main peak frequency (Fmax)")
    peaks, _ = find_peaks(PSD)
    return max(PSD[peaks])


### Mean Power
def mp(PSD):
    """

    :param PSD:
    :return:
    """
    logger.info("Mean Power (MP)")
    #    f, Pxx_den = periodogram(L, fs)
    return PSD.mean()


### Total Power
def tp(PSD):
    """

    :param PSD:
    :return:
    """
    logger.info("Total Power (TP)")
    return PSD.sum()


### Mean Frequency (MNF)
def meanfreq(x, fs=100.0, secs=4):
    """
    Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
    Estimates the mean normalized frequency of the power spectrum

    :param x:
    :param fs:
    :param secs:
    :return:
    """
    logger.info("Mean Frequency (MNF)")
    win = secs * fs
    f, Pxx_den = welch(x, fs, nperseg=win)
    Pxx_den = np.reshape(Pxx_den, (1, -1))
    width = np.tile(f[1] - f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    pwr = np.sum(P)
    mnfreq = np.dot(P, f.T) / pwr
    return mnfreq[0, 0]


### Median frequency (MDF)
# Estimates the median normalized frequency of the power spectrum
def medfreq(x, fs=100.0, secs=4):
    """
    Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
    Estimates the median normalized frequency of the power spectrum
    :param x:
    :param fs:
    :param secs:
    :return:
    """
    logger.info("Median Frequency (MDF)")
    win = secs * fs
    f, Pxx_den = welch(x, fs, nperseg=win)
    Pxx_den = np.reshape(Pxx_den, (1, -1))
    width = np.tile(f[1] - f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    return np.median(P)


### Standard Sesviation of the power (std)
def std_psd(Pxx_den):
    """

    :param Pxx_den:
    :return:
    """
    logger.info("Standard Sesviation of the power (std)")
    return Pxx_den.std()


### 1st, 2nd, 3rd spectral moments (SM1, SM2, SM3)
def mmnt(Pxx_den, order=1):
    """

    :param Pxx_den:
    :param order:
    :return:
    """
    logger.info("1st, 2nd, 3rd spectral moments (SM1, SM2, SM3)")
    return moment(Pxx_den, moment=order)


def kurt(Pxx_den):
    """

    :param Pxx_den:
    :return:
    """
    logger.info("Kurtosis of the power spectrum")
    return kurtosis(Pxx_den)


def skw(Pxx_den):
    """

    :param Pxx_den:
    :return:
    """
    logger.info("Skewness of the power spectrum")
    return skew(Pxx_den)


### Autocorrelate
def autocorr(L):
    """
    3 firsts coefficients of the autocorrelation
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    :param L:
    :return:
    """
    result = np.correlate(L, L, mode='full')
    return result[result.size // 2:][:3]
