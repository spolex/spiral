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
from math import log, pi, degrees
from functools import partial
import time
from pyentrp import entropy as e

import logging

logger = logging.getLogger('MainLogger')


def idct_vec(n, Xk, c, N):
    """
    idct auxiliar function
    :param n: scalar
    :param Xk: 1-N dimensional array
    :param l: 1-N dimensional array
    :param N: scalar
    :return:
    """
    start_time = time.time()
    k = np.arange(c)
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


def idct(Xk, c=17):
    """
    Inverse Discrete Cosine Transformation
    :param Xk: 1-N dimensional array
    :param l: scalar number of coefficients
    :return: 1-N dimensional array
    """
    start_time = time.time()
    N = len(Xk)

    Xk = Xk[:c]
    n = np.arange(N)
    X = np.array(list(map(partial(idct_vec, Xk=Xk, c=c, N=N), n)))
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate idct  is %s", elapsed_time)
    return X


def dct_vect(k, N, X):
    """
    Discrete Cosine Transformation auxiliar function
    :param k: scalar
    :param N: scalar
    :param X: 1-N dimensional array
    :return: 1-N dimensional array
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
    Discrete Cosine Transformation
    :param X: 1N dimensional array
    :return: 1N dimensional array
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
    Radius (euclidean distance)
    :param x: x coordinate
    :param y: y coordinate
    :return:
    """
    xc = x[0]
    yc = y[0]

    start_time = time.time()
    r = np.sqrt((x-xc)**2+(y-yc)** 2)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate radius(x,y) is %s", elapsed_time)
    return r


def residues(x, c=None):
    """
    # TODO https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
    :param x: 1-N dimensional array
    :param l: number of coefficients to reconstruct origonal array
    :return: 1-N dimensional array
    """
    start_time = time.time()
    idct_x = idct(dct(x), c=c)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate residuos is %s", elapsed_time)
    return x - idct_x


def cart2pol(x, y):
    """
    From cartesian coordinates to polar coordinates
    :param x: x coordinate
    :param y: y coordinate
    :return: polar coordinates (rho, phi)
    """
    start_time = time.time()
    rho = radio(x, y)
    phi = np.arctan2(y, x)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to transform from cartesian coordinates to polar coordinates is %s", elapsed_time)
    return rho, np.array(list(map(degrees,phi)))


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


def samp_ent(x, m=3, r=0.2):
    """
    https://sampen.readthedocs.io/en/stable/#documentation
    :param r:
    :param m:
    :param x: time series data
    :return: sample entropy
    """
    start_time = time.time()
    se = sampen2(x, mm=m, r=r)
    elapsed_time = time.time() - start_time
    rdo = se[len(se) - 1][1]
    logger.info("Sample Entropy %s", rdo)
    logger.debug("Elapsed time to calculate sample entropy is %s", elapsed_time)
    return rdo


def mean_abs_val(L):
    """
    Mean Absolute Value (MAV)
    :param L: 1-N dimensional array
    :return: scalar
    """
    start_time = time.time()
    N = len(L)
    mav = (1 / N) * sum(abs(L))
    elapsed_time = time.time() - start_time
    logger.info("Mean Absolute Value (MAV) %s", mav)
    logger.debug("Elapsed time to calculate mean absolute value is %s", elapsed_time)
    return mav


def diff_abs_std(L):
    """
    Difference Absolute standard deviation (AAC)
    :param L: 1-N dimensional array
    :return: scalar
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
    logger.info("Difference Absolute standard deviation (AAC) %s", aac)
    return aac


def log_detector(L):
    """
    Log Detector (LD)
    :param L: 1-N dimesional array
    :return: scalar
    """
    start = time.time()
    N = len(L)
    f = (1 / N) * np.log(np.abs(L)).sum()
    ld = np.nanstd(L) ** (f)
    elapsed_time = time.time() - start
    logger.info("Log Detector (LD) %s", ld)
    logger.debug("Elapsed time to calculateLog Detector (LD) is %s", elapsed_time)
    return ld


def wl(L):
    """
    Waveform length (WL)
    :param L:
    :return:
    """
    start = time.time()
    w = sum(abs(np.diff(L)))
    elapsed_time = time.time() - start
    logger.info("Waveform length (WL) %s", w)
    logger.debug("Elapsed time to calculate Waveform length (WL) is %s", elapsed_time)
    return w


def root_mean_square(L):
    """
    Root Mean Square
    :param L: 1-N dimensional array
    :return: scalar
    """
    start = time.time()
    N = len(L)
    rme = ((1 / N) * L ** 2).sum()
    elapsed_time = time.time() - start
    logger.info("Root Mean Square %s", rme)
    logger.debug("Elapsed time to calculate Root Mean Square is %s", elapsed_time)
    return rme


def higuchi(L):
    """
    Higuchi's Fractal Dimension (hfd)
    :param L:
    :return:
    """
    start_time = time.time()
    h = hfda.measure(L, 5)
    elapsed_time = time.time() - start_time
    logger.debug("Elapsed time to calculate higuchi fractal dimension is %s", elapsed_time)
    logger.info("Higuchi's Fractal Dimension %s", h)
    return h


def wamp(l):
    """
    Wilson amplitude (WAMP)
    :param l: 1-N dimensional array
    :return: scalar
    """
    start = time.time()
    epsilon = l.mean()
    diff = abs ( l.diff().dropna().values )
    amp = diff > epsilon
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Wilson amplitude (WAMP) value is %s", elapsed_time)
    wampli = sum(amp)
    logger.info("Wilson amplitude (WAMP) %s", wampli)
    return wampli


def mfl(l):
    """
    Maximum fractal length (MFL)
    :param l: 1N dimensional array
    :return: scalar
    """
    start = time.time()
    mxfl = log(wl(l))
    elapsed_time = time.time() - start
    logger.info("Maximum fractal length (MFL) %s", mxfl)
    logger.debug("Elapsed time to calculate Maximum fractal length (MFL) value is %s", elapsed_time)
    return mxfl


def myo(L):
    """
    Myopulse percentage rate (MYO) Percentage of time where the signal is bigger than two times the mean
    :param L:
    :param ts:
    :return:
    """
    start = time.time()
    N = len(L)
    dos_epsilon = 2 * L.mean()
    biggers = L[L > dos_epsilon]
    my = len(biggers) / N
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Myopulse percentage rate (MYO) value is %s", elapsed_time)
    logger.info("Myopulse percentage rate (MYO) %s", my)
    return my


def iemg(L):
    """
    Integrated EMG (IEMG)
    :param L:
    :return:
    """
    start = time.time()
    mg = abs(L).sum()
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Integrated EMG (IEMG) value is %s", elapsed_time)
    logger.info("Integrated EMG (IEMG) %s", mg)
    return mg


def ssi(L):
    """
    Simple square EMG (SSI)
    :param L:
    :return:
    """
    start = time.time()
    rdo = (L ** 2).sum()
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Simple square EMG (SSI) value is %s", elapsed_time)
    logger.info("Simple square EMG (SSI) %s", rdo)
    return rdo


def zc(l, epsilon=None):
    """
    Zero crossing (ZC): The number of times in which the signal crosses its mean
    :param l: 1-N dimensional array
    :param epsilon: crossing barrier, if None the mean of l is being calculated
    :return: scalar Zero crossing (ZC)
    """
    start = time.time()

    if epsilon is None:
        epsilon = l.mean()
    c = len(l[l > epsilon])
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Zero crossing (ZC) value is %s", elapsed_time)
    logger.info("Zero crossing (ZC) %s", c)
    return c


def masks(l):
    """

    :param l: 1-N dimensional array
    :return:
    """
    d = np.diff(l).round()
    dd = np.diff(d).round()
    # Mask of locations where graph goes to vertical or horizontal, depending on vec
    to_mask = ((d[:-1] != 0) & (d[:-1] == -dd))
    # Mask of locations where graph comes from vertical or horizontal, depending on vec
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask


def ssc(l):
    """
    Slope sign change (SSC)
    :param l: 1-N dimensional array
    :return: scalar representing Slope Sign Change (SSC)
    """
    start = time.time()
    to_vert_mask, from_vert_mask = masks(l)
    ss = sum(to_vert_mask) + sum(from_vert_mask)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Slope sign change (SSC) value is %s", elapsed_time)
    logger.info("Slope sign change (SSC) %s", ss)
    return ss


def p_max(psd, l):
    """
    Maximum peak of frequency (Pmax)
    :param psd: Estimate power spectral density
    :param l: Origin data of psd
    :return:
    """
    start = time.time()
    peaks, _ = find_peaks(psd)
    pmx = max(l[peaks])
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Maximum peak of frequency (Pmax) value is %s", elapsed_time)
    logger.info("Main peak amplitude (Pmax) %s", pmx)
    return pmx


def f_max(psd):
    """
    Frequency of the max peak
    :param psd:
    :return:
    """
    start = time.time()
    peaks, _ = find_peaks(psd)
    fmx = max(psd[peaks])
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Main peak frequency (Fmax) value is %s", elapsed_time)
    logger.info("Main peak frequency (Fmax) %s", fmx)
    return fmx


def mp(psd):
    """
    Mean Power (MP)
    :param psd:
    :return:
    """
    start = time.time()
    mp = psd.mean()
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Mean Power (MP) value is %s", elapsed_time)
    logger.info("Mean Power (MP) %s", mp)
    return mp


def tp(psd):
    """
    Total Power (TP)
    :param psd:
    :return:
    """
    start = time.time()
    s = psd.sum()
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Total Power (TP) value is %s", elapsed_time)
    logger.info("Total Power (TP) %s", s)
    return s


def meanfreq(x, fs=100.0, secs=4):
    """
    Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
    Estimates the mean normalized frequency of the power spectrum

    :param x:
    :param fs:
    :param secs:
    :return:
    """
    start = time.time()
    win = secs * fs
    f, pxx_den = welch(x, fs, nperseg=win)
    pxx_den = np.reshape(pxx_den, (1, -1))
    width = np.tile(f[1] - f[0], (1, pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = pxx_den * width
    pwr = np.sum(P)
    mnfreq = np.dot(P, f.T) / pwr
    elapsed_time = time.time() - start
    logger.info("Mean Frequency (MNF) %s", mnfreq[0, 0])
    logger.debug("Elapsed time to calculate Mean Frequency (MNF) value is %s", elapsed_time)

    return mnfreq[0, 0]


def medfreq(x, fs=100.0, secs=4):
    """
    Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
    Estimates the median normalized frequency of the power spectrum
    :param x:
    :param fs:
    :param secs:
    :return:
    """
    start = time.time()
    win = secs * fs
    f, pxx_den = welch(x, fs, nperseg=win)
    pxx_den = np.reshape(pxx_den, (1, -1))
    width = np.tile(f[1] - f[0], (1, pxx_den.shape[1]))
    P = pxx_den * width
    median = np.median(P)
    elapsed_time = time.time() - start
    logger.info("Median Frequency (MNF) %s", median)
    logger.debug("Elapsed time to calculate Median Frequency (MDF) value is %s", elapsed_time)
    return median


def std_psd(pxx_den):
    """
    Standard Sesviation of the power (std)
    :param Pxx_den:
    :return:
    """
    start = time.time()
    s = pxx_den.std()
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Standard Sesviation of the power (std) value is %s", elapsed_time)
    logger.info("PSD Standard Desviation (std) %s", s)
    return s


def mmnt(Pxx_den, order=1):
    """
    1st, 2nd, 3rd spectral moments (SM1, SM2, SM3)
    :param Pxx_den:
    :param order:
    :return:
    """
    start = time.time()
    mm = moment(Pxx_den, moment=order)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate moment value is %s", elapsed_time)
    logger.info("%s spectral moment (SM1, SM2, SM3) %s", order, mm)
    return mm


def kurt(Pxx_den):
    """
    Kurtosis of the power spectrum
    :param Pxx_den:
    :return:
    """
    start = time.time()
    kt = kurtosis(Pxx_den)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate kurtosis value is %s", elapsed_time)
    logger.info("Kurtosis of the power spectrum %s", kt)
    return kt


def skw(Pxx_den):
    """
    Skewness of the power spectrum
    :param Pxx_den:
    :return:
    """
    start = time.time()
    sk = skew(Pxx_den)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate skewness value is %s", elapsed_time)
    logger.info("Skewness of the power spectrum %s", sk)
    return sk


def autocorr(L):
    """
    3 firsts coefficients of the autocorrelation
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    :param L:
    :return:
    """
    start = time.time()
    result = np.correlate(L, L, mode='full')
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate autorrelation value is %s", elapsed_time)
    return result[result.size // 2:][:3]


def derivative(l, order=1):
    """
    The gradient is computed using second order accurate central differences in the interior points and either first
    or second order accurate one-sides (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.
    :param l:
    :param order:
    :return:
    """
    start = time.time()
    logger.info("Derivative of order %s", order)
    gr = np.gradient(l, edge_order=order)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate derivative value is %s", elapsed_time)
    return gr


def shannon_entrp(l):
    """
    Shannon entropy
    :param l:
    :return:
    """
    start = time.time()
    se = e.shannon_entropy(l)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Shannon entropy value is %s", elapsed_time)
    return se


def perm_entrp(l):
    """
    Permutation entropy
    :param l:
    :return:
    """
    start = time.time()
    se = e.permutation_entropy(l)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Permutation entropy value is %s", elapsed_time)
    return se


def multi_entrp(l):
    """
    Multiscale entropy
    :param l:
    :return:
    """
    start = time.time()
    me = e.multiscale_entropy(l)
    elapsed_time = time.time() - start
    logger.debug("Elapsed time to calculate Multiscale entropy value is %s", elapsed_time)
    return me
