from datetime import date
from os import path
from scipy.signal import resample
from pandas import pandas as pd
import tensorflow as tf
import numpy as np

results_path = "/data/elekin/data/results/handwriting"

def load_raw_data(doc_path, filename, features, cols, n_classes=3):
    """ """
    meta_df = pd.read_csv(path.join(doc_path, filename), index_col=0)
    x_train = []
    y_train = []

    labels = None
    if n_classes > 1:
        labels = meta_df.level
    else:
        labels = meta_df.temblor.astype(np.int16)

    for file_path, level in zip(meta_df.abs_path, labels):
        df = pd.read_csv(file_path, sep="\s+", header=None, names=features, skiprows=1, usecols=cols)
        x_train.append(resample(df.values.astype('int16'), 4096))
        y_train.append(level)
    return x_train, y_train

def load_from_csv(columns, today=date.today().strftime("%Y%m%d"), level=False, n=4096):

    features = pd.read_csv(path.join(results_path,"biodarw_{}.csv".format(today))).set_index('subject_id')
    labels = []
    y = []
    if level:
        levels = pd.read_csv(path.join(results_path,"level_{}.csv".format(today))).set_index('subject_id')
        y = levels.values.astype(np.int16).ravel()
    else:
        labels = pd.read_csv(path.join(results_path,"binary_labels_{}.csv".format(today))).set_index('subject_id')
        y = (labels == 'si').values.astype(np.int16).ravel()

    df_rs = features[columns].groupby('subject_id').apply(resample, n)
    X = np.array(df_rs.values.tolist())
    
    return X, y


    

