# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:08:24 2019

@author: isancmen
"""

import os
import numpy as np
from loaders.reader_and_writer import load, save
from properties.properties import Properties
from skrebate import ReliefF
from sklearn.feature_selection import RFE


def dataset_prep(filename, filename_ds, mode='r'):
    """

    :param mode:
    :param filename:
    :param filename_ds:
    :return:
    """
    controls = Properties.controls
    et = Properties.et
    ct_labels = np.ones(controls).astype(np.dtype('>i4'))
    et_labels = np.full(et, 2).astype(np.dtype('>i4'))

    if os.path.exists(filename):

        # labels
        labels = np.hstack((ct_labels, et_labels))

        # union residues features ET and CT
        df_rd_ct = load(filename, Properties.rd_ct, mode)
        df_rd_et = load(filename, Properties.rd_et, mode)
        df_rd_ct_et = np.vstack((df_rd_ct, df_rd_et))

        # union residues features ET and CT
        df_rd_ct_fe = load(filename, Properties.rd_ct_fe, mode)
        df_rd_et_fe = load(filename, Properties.rd_et_fe, mode)
        df_rd_ct_et_fe = np.vstack((df_rd_ct_fe, df_rd_et_fe))

        # feature selection
        fltr = RFE(ReliefF(), n_features_to_select=5, step=0.5)
        filtered_df_rd_et_fe = fltr.fit_transform(df_rd_ct_et_fe, labels)

        # concatenate residues and selected fetures
        train_rd = np.hstack((df_rd_ct_et, filtered_df_rd_et_fe))

        # Save results
        save(filename_ds, Properties.train_rd, train_rd)

        # union radius ET and CT
        df_r_ct = load(filename, Properties.r_ct, mode)
        df_r_et = load(filename, Properties.r_et, mode)
        df_r_ct_et = np.vstack((df_r_ct, df_r_et))

        # union radius features ET and CT
        df_r_ct_fe = load(filename, Properties.r_ct_fe, mode)
        df_r_et_fe = load(filename, Properties.r_et_fe, mode)
        df_r_ct_et_fe = np.vstack((df_r_ct_fe, df_r_et_fe))

        # feature selection
        filtered_df_r_et_fe = fltr.fit_transform(df_r_ct_et_fe, labels)

        # concatenate radius and selected fetures
        train_r = np.hstack((df_r_ct_et, filtered_df_r_et_fe))

        # Save results
        save(filename_ds, Properties.train_r, train_r)
        save(filename_ds, Properties.labels, labels)

    else:
        print(filename + " doesn't exist")


filepath = "../output/archimedean-17.h5"
filepath_ds = "../output/archimedean_ds-17.h5"
dataset_prep(filepath, filepath_ds)
