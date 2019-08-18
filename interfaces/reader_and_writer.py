# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:56:46 2019

@author: isancmen
"""

import h5py
import logging

logger = logging.getLogger('MainLogger')

def save(filename, datasetname, data):
    try:
        hf = h5py.File(filename, 'r+')
        del hf[datasetname]
    except OSError:
        logger.warning(filename+" doesn't exist")
        hf = h5py.File(filename, 'w')
        pass
    except KeyError:
        logger.warning("Dataset "+ datasetname + " doesn't exist")
        hf = h5py.File(filename, 'r+')
        pass
    finally:
        logger.debug("Creating dataset "+ datasetname + " into file " + filename)
        hf[datasetname] = data
        hf.close()

def load(filename, datasetname, mode='r'):
    hf = h5py.File(filename, mode)
    dataset = hf[datasetname][:]
    hf.close()
    return dataset