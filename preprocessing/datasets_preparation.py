# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:08:24 2019

@author: isancmen
"""

import os
import numpy as np
from interfaces.reader_and_writer import load,save

filename="../output/archimedean.h5"
filename_ds="../output/archimedean_ds.h5"
##filename_labels="../output/archimedean.h5"


controls=27
et=23
ct_rd='rd_ct_fe'
ct_r='r_ct_fe'
ct_labels = np.ones(27).astype(np.dtype('>i4') )
et_rd='rd_et_fe'
et_r='r_et_fe'
et_labels = np.full(23,2).astype(np.dtype('>i4') )
mode= 'r'

def dataset_prep():
    if os.path.exists(filename):
        df_rd = load(filename,ct_rd,mode)
        df_rd_et = load(filename,et_rd,mode)
        train_rd = np.vstack((df_rd,df_rd_et))
        save(filename_ds, 'train_rd', train_rd)
        df_r = load(filename,ct_r,mode)
        df_r_et = load(filename,et_r,mode)
        train_r = np.vstack((df_r,df_r_et))
        save(filename_ds, 'train_r', train_r)
        labels = np.hstack((ct_labels,et_labels))
        save(filename_ds, 'labels', labels)
        
dataset_prep()