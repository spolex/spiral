# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:41:36 2019

@author: isancmen
"""
from scipy.fftpack import dct, idct
from entropy import sample_entropy,higuchi_fd
import numpy as np
from scipy.signal import find_peaks,periodogram,welch
from scipy.stats import moment,kurtosis,skew
from math import log

import logging

logger = logging.getLogger('MainLogger')

def radio(x,y):
    return (x**2+y**2)**(1/2)

def residuos(x):
    idct_x = idct(dct(x, norm='ortho'), norm='ortho')
    return x-idct_x

# sample entropy
def samp_ent(r):
    return sample_entropy(r, order=3, metric='chebyshev')

# mean absolute value
def mean_abs_val(L):
    N = len(L)
    return (1/N)*L.sum()

### Difference Absolute standard deviation (AAC)
def diff_abs_std(L):
    N=len(L)
    logger.debug("Timeseries' max is %s", max(L))
    den=1/(N-1)
    square_diff=(np.diff(L))**2        
    logger.debug("Square differences max %s", max(square_diff))
    avg_diff=den*sum(square_diff)
    logger.debug("Avg differences is %s", avg_diff)
    return avg_diff**(1/2)

## log detector
def log_detector(L):
    return L.std()**(mean_abs_val(L))

### Waveform length (WL)
def wl(L):
    return sum(abs(np.diff(L)))

### Root mean square
def root_mean_square(L):
    N = len(L)
    return ((1/N)*L**2).sum()

### higuchi fractal dimension
def higuchi(L):
    logger.info("Calculation of Fractal Dimension")
    return higuchi_fd(L, kmax=5)

### Wilson amplitude (WAMP)    
def wamp(L):
    epsilon=L.mean()
    return np.array(abs(L[:-1] - L [1:])>epsilon,dtype=int).sum()  

### Maximum fractal length (MFL)
def mfl(L):
    logger.info("Calculation of Maximum fractal length (MFL)")
    return log(wl(L)) 

### Myopulse percentage rate (MYO)
def myo(L,ts):
    """
    Description: Percentage of time where the signal is bigger than two times the mean
    """
    assert len(ts) > 0 ,"Myopulse percentage rate needs timestamp"
    logger.info("Calculation of Myopulse percentage rate (MYO)")
    N = len(L)
    dos_epsilon = 2*L.mean()
    biggers =L[ L>dos_epsilon ]
    logger.debug("Myopulse times rate (MYO) %s",len(biggers))
    myo=len(biggers)/N
    logger.debug("Myopulse percentage rate (MYO) %s",myo)
    return myo

### Integrated EMG (IEMG)
def iemg(L):
    logger.info("Calculation of Integrated EMG (IEMG)")
    return abs(L).sum()

### Simple square EMG (SSI)
def ssi(L):
    return (L**2).sum()

### Zero crossing (ZC)
def zc(L, epsilon=None):
    """
    Description: The number of times in which the signal crosses its mean
    """
    if(epsilon == None):
        epsilon = L.mean()
    return len(L[L>epsilon])

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
    to_vert_mask, from_vert_mask = masks(L)
    return sum(to_vert_mask)+sum(from_vert_mask)
    
### Mean Frequency (MNF)
def meanfreq(x, fs=100.0, secs=4):
    """
    fs: float
        Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
        Estimates the mean normalized frequency of the power spectrum
    """
    win = secs * fs
    f, Pxx_den = welch(x, fs, nperseg=win)                                                    
    Pxx_den = np.reshape(Pxx_den, (1,-1) ) 
    width = np.tile(f[1]-f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    pwr = np.sum(P)
    mnfreq = np.dot(P, f.T)/pwr
    return mnfreq

### Main peak amplitude (Pmax)
def p_max(L):
    peaks, _ = find_peaks(L)
    return max(L[peaks])

### Main peak frequency (Fmax)
def f_max(L, fs=100.0):
    """
    //TODO implementar numpy
    Frequency of the max peak
    """
    f, Pxx_den = periodogram(L, fs) 
    peaks, _ = find_peaks(Pxx_den)
    return Pxx_den[np.where(L == max(L[peaks]))][0]

### Mean Power
def mp(L,fs=100.0):
    f, Pxx_den = periodogram(L, fs) 
    return Pxx_den.mean()

### Total Power
def tp(L,fs=100.0):
    f, Pxx_den = periodogram(L, fs) 
    return Pxx_den.sum()

### Meadian frequency (MDF)
#Estimates the median normalized frequency of the power spectrum
def medfreq(x, fs=100.0, secs=4):
    """
    fs: float
        Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
        Estimates the median normalized frequency of the power spectrum
    """
    win = secs * fs
    f, Pxx_den = welch(x, fs, nperseg=win)                                                    
    Pxx_den = np.reshape(Pxx_den, (1,-1) ) 
    width = np.tile(f[1]-f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    return P[int(len(P)/2)]

### Standard Sesviation of the power (std)
def std_psd(L,fs=100.0):
    f, Pxx_den = periodogram(L, fs) 
    return Pxx_den.std()

### 1st, 2nd, 3rd spectral moments (SM1, SM2, SM3)
def mmnt(L,order=1,fs=100.0):
    f, Pxx_den = periodogram(L, fs) 
    return moment(Pxx_den, moment=order)

def kurt(L,fs=100.0):
    f, Pxx_den = periodogram(L, fs) 
    return kurtosis(Pxx_den)

def skw(L,fs=100.0):
    f, Pxx_den = periodogram(L, fs) 
    return skew(Pxx_den)

### Autocorrelate
def autocorr(L):
    """
    3 firsts coefficients of the autocorrelation
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    """
    result = np.correlate(L, L, mode='full')
    return result[result.size // 2:][:3]