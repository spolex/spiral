# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:56:46 2019

@author: isancmen
"""

from loaders import h5py, logging

logger = logging.getLogger('MainLogger')


def save(filename, datasetname, data):
    """
    Save a dataset "datasetname" to "filename" hdf5 file
    :param filename: string
    :param datasetname: string
    :param data: array
    :return: void
    """
    try:
        hf = h5py.File(filename, 'r+')
        del hf[datasetname]
    except OSError:
        logger.warning(filename + " doesn't exist")
        hf = h5py.File(filename, 'w')
        pass
    except KeyError:
        logger.warning("Dataset " + datasetname + " doesn't exist")
        hf = h5py.File(filename, 'r+')
        pass
    finally:
        logger.debug("Creating dataset " + datasetname + " into file " + filename)
        hf[datasetname] = data
        hf.close()


def load(filename, datasetname, mode='r'):
    """
    Load a dataset "datasetname" from "filename" hdf5 file
    :param filename: string
    :param datasetname: string
    :param mode:
    :return: hdf5 dataset
    """
    hf = h5py.File(filename, mode)
    dataset = hf[datasetname][:]
    hf.close()
    return dataset
