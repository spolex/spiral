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
from math import log,cos,pi

import logging

logger = logging.getLogger('MainLogger')

def idct(Xk,l=None):
    N = len(Xk)
    if l != None:
        assert N > l, "l can not be bigger than len of Xk"
    l = l if l != None else N
    
    Xk = Xk[:l]
    X = np.zeros(N)
    for n in range(N):
        k = np.arange(l)
        c = np.zeros(k.shape)
        c[0] = (1/N)**(1/2)
        c[1:] = (2/N)**(1/2)
        f = (pi/N)*(n+(1/2))*k
        logger.debug("Length of array function is %s",len(f))
        cs = np.cos(f)
        logger.debug("Length of array cosin is %s",len(cs))
        xn = c*Xk*cs
        logger.debug("Length of xn is %s",len(xn))
        rdo = xn.sum()
        logger.debug("Value %s is %s", n, rdo)
        X[n]=rdo
    return X

def dct(X):
    N = len(X)
    
    Xk = np.zeros(X.shape)
    for k in range(N):
        n = np.arange(N)
        c = (1/N)**(1/2) if k == 0 else (2/N)**(1/2)
        f = (pi/N)*(n+(1/2))*k
        logger.debug("Length of array function is %s",len(f))
        cs = np.cos(f)
        logger.debug("Length of array cosin is %s",len(cs))
        xk = c*X*cs
        logger.debug("Length of xk is %s",len(xk))
        Xk[k]=xk.sum()
    return Xk
    

def radio(x,y):
    return (x**2+y**2)**(1/2)

def residuos(x,coef=17):
    # TODO https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html 
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
    logger.info("higuchi Fractal Dimension")
    return higuchi_fd(L, kmax=5)

### Wilson amplitude (WAMP)    
def wamp(L):
    logger.info("Wilson amplitude (WAMP)")
    epsilon=L.mean()
    a=L[:-1].values
    b=L[1:].values
    diff=abs(a-b)
    amp=diff>epsilon
    return amp.sum()  

### Maximum fractal length (MFL)
def mfl(L):
    logger.info("Maximum fractal length (MFL)")
    return log(wl(L)) 

### Myopulse percentage rate (MYO)
def myo(L,ts):
    """
    Description: Percentage of time where the signal is bigger than two times the mean
    """
    assert len(ts) > 0 ,"Myopulse percentage rate needs timestamp"
    logger.info("Myopulse percentage rate (MYO)")
    N = len(L)
    dos_epsilon = 2*L.mean()
    biggers =L[ L>dos_epsilon ]
    logger.debug("Myopulse times rate (MYO) %s",len(biggers))
    myo=len(biggers)/N
    logger.debug("Myopulse percentage rate (MYO) %s",myo)
    return myo

### Integrated EMG (IEMG)
def iemg(L):
    logger.info("Integrated EMG (IEMG)")
    return abs(L).sum()

### Simple square EMG (SSI)
def ssi(L):
    logger.info("Simple square EMG (SSI)")
    return (L**2).sum()

### Zero crossing (ZC)
def zc(L, epsilon=None):
    """
    Description: The number of times in which the signal crosses its mean
    """
    logger.info("Zero crossing (ZC)")
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
    logger.info("Slope sign change (SSC)")
    to_vert_mask, from_vert_mask = masks(L)
    return sum(to_vert_mask)+sum(from_vert_mask)
    
### Main peak amplitude (Pmax)
def p_max(psd,L):
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
    logger.info("Mean Power (MP)")
#    f, Pxx_den = periodogram(L, fs) 
    return PSD.mean()

### Total Power
def tp(PSD):
    logger.info("Total Power (TP)")
    return PSD.sum()

### Mean Frequency (MNF)
def meanfreq(x, fs=100.0, secs=4):
    """
    fs: float
        Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
        Estimates the mean normalized frequency of the power spectrum
    """
    logger.info("Mean Frequency (MNF)")
    win = secs * fs
    f, Pxx_den = welch(x, fs, nperseg=win)                                                    
    Pxx_den = np.reshape(Pxx_den, (1,-1) ) 
    width = np.tile(f[1]-f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    pwr = np.sum(P)
    mnfreq = np.dot(P, f.T)/pwr
    return mnfreq[0,0]

### Median frequency (MDF)
#Estimates the median normalized frequency of the power spectrum
def medfreq(x, fs=100.0, secs=4):
    """
    fs: float
        Sampling frequency of the x time series in units of Hz. Defaults to 100.0.
        Estimates the median normalized frequency of the power spectrum
    """
    logger.info("Median Frequency (MDF)")
    win = secs * fs
    f, Pxx_den = welch(x, fs, nperseg=win)                                                    
    Pxx_den = np.reshape(Pxx_den, (1,-1) ) 
    width = np.tile(f[1]-f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    return np.median(P)

### Standard Sesviation of the power (std)
def std_psd(Pxx_den):
    logger.info("Standard Sesviation of the power (std)")
    return Pxx_den.std()

### 1st, 2nd, 3rd spectral moments (SM1, SM2, SM3)
def mmnt(Pxx_den,order=1):
    logger.info("1st, 2nd, 3rd spectral moments (SM1, SM2, SM3)")
    return moment(Pxx_den, moment=order)

def kurt(Pxx_den):
     logger.info("Kurtosis of the power spectrum")
     return kurtosis(Pxx_den)

def skw(Pxx_den):
     logger.info("Skewness of the power spectrum")
     return skew(Pxx_den)

### Autocorrelate
def autocorr(L):
    """
    3 firsts coefficients of the autocorrelation
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    """
    result = np.correlate(L, L, mode='full')
    return result[result.size // 2:][:3]