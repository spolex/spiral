#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

from os import path
import pathlib


from pandas import HDFStore
import pandas as pd
import math
import numpy as np
from scipy.signal import resample

import matplotlib as mpl
import matplotlib.pyplot as plt

logdir = pathlib.Path(".spiral")/"tensorboard_logs"
features = ['x', 'y', 'timestamp', 'pen_up', 'azimuth', 'altitude', 'pressure']


#Early stop configuration
earlystop_callback = EarlyStopping(
  monitor='val_accuracy', min_delta=1e-2,
  patience=200)

training_earlystop_callback = EarlyStopping(
  monitor='accuracy', min_delta=1e-2,
  patience=200)
    
def load_residues(root_path, num_coefficients):
    print(num_coefficients)
    h5file = path.join(root_path, "archimedean-")
    h5filename = h5file + str(num_coefficients) + ".h5"
    print("Loading file {0}".format(h5filename))
    hdf = HDFStore(h5filename)
    raw_df = hdf['results/residues/rd'].T
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_df = pd.DataFrame(scaler.fit_transform(raw_df))
    scaled_df['target'] = hdf.get('source/labels').values
    hdf.close()
    return scaled_df

def load_residues_dataset(root_path, num_coefficients):
    scaled_df=load_residues(root_path, num_coefficients)
    target=scaled_df.pop("target")
    return tf.data.Dataset.from_tensor_slices((scaled_df.values.astype('float32'), 
                                              target.values.astype('int8').reshape(-1,1)))

"""Training helpper functions
"""

"""Plot accuracy for train and val datasets and also loss function vs number of epoch
"""
def plot_report(histories, metric='accuracy'):
    legend_pairs = [['Train_{}'.format(key), 'Val_{}'.format(key)] for key in histories]
    legend = [item for sublist in legend_pairs for item in sublist]
    
    for key in histories:
        history=histories[key]
        # Plot training & validation loss values
        plt.plot(history.history[metric])
        plt.plot(history.history['val_{}'.format(metric)][1:], '--')

    plt.title('Model {}'.format(metric))
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    
def plot_report_t(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


    
"""Build basic lstm architecture
"""
def build_basic_lstm(num_cells, size, normalize=True, dropout=0.1, num_classes=1): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=(size,)))
    model.add(tf.keras.layers.LSTM(num_cells, activation=tf.nn.tanh, return_sequences=False))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.sigmoid))
    return model

"""build CNN architecture from paper Parkinson’s Disease Detection from Drawing
Movements Using Convolutional Neural Networks
Manuel Gil-Martín , Juan Manuel Montero and Rubén San-Segundo
"""
def build_mdpi(size, dropout=0.5):
    return tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),# expand the dimension form (50, 4096) to (50, 4096, 1)
                          input_shape=(size,)),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Conv1D(filters=16, kernel_size=5,
                          strides=1, padding="causal",
                          activation=tf.nn.relu),
        tf.keras.layers.MaxPool1D(pool_size=3),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Conv1D(filters=16, kernel_size=5,
                          strides=1, padding="causal",
                          activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(32, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,kernel_regularizer=tf.keras.regularizers.l2(0.0001))])

def build_cnn(size, dropout=0.5):
    return tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),# expand the dimension form (50, 4096) to (50, 4096, 1)
                          input_shape=(size,)),
        tf.keras.layers.Conv1D(filters=16, kernel_size=5,
                          strides=1, padding="causal",
                          activation=tf.nn.relu),
        tf.keras.layers.MaxPool1D(pool_size=3),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Conv1D(filters=16, kernel_size=5,
                          strides=1, padding="causal",
                          activation=tf.nn.relu),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])


"""Data preprocessing functions
"""
def windowed_dataset(series, window_size, batch_size,label):
    """Create a windowed dataset from numpy series using tf.Data.Dataset using from_tensor_slices method
    """
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size))
    #ds = ds.shuffle(shuffle_buffer).map(lambda window: (window[:-1], label)) No need shuffle because is the same label each row
#     ds = ds.map(lambda window: (window[:-1], label)) # regression splits the window to use the last value (-1) as label
    ds = ds.map(lambda window: (window, label))
    #ds = ds.batch(batch_size).prefetch(1) # better batch after concatenate and shuffle all the windowed samples
    return ds


def split_train_test_val(full_dataset, data_size, train_ratio, val_ratio, test_ratio, shuffle_buffer=16e6, seed=42, mini_batch_size=4):
    """Create dataset splits for training pruposes from tensorflow dataset full_dataset
    """
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    test_size = int(test_ratio * data_size)

    full_dataset = full_dataset.shuffle(shuffle_buffer, seed=seed)
    train_dataset = full_dataset.take(train_size).batch(mini_batch_size).prefetch(2).cache()
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size).batch(mini_batch_size).prefetch(2).cache()
    test_dataset = test_dataset.take(test_size).batch(mini_batch_size).prefetch(2).cache()
    return (train_dataset, val_dataset, test_dataset)

def split_train_test(full_dataset, ratio=0.67, seed=42):
    """Create dataset splits for training pruposes from tensorflow dataset full_dataset
    """
    size= math.ceil(len(full_dataset)*ratio)
    full_dataset = full_dataset.shuffle(len(full_dataset), seed=seed)
    train_dataset = full_dataset.take(size).batch(1).prefetch(4).cache()
    test_dataset = full_dataset.skip(size).batch(1).prefetch(4).cache()
    
    return (train_dataset, test_dataset)

def get_abs_path(df, registro_tableta, idx=0):
    return df[(df.index == registro_tableta)].abs_path

def read(filename):
    df = pd.read_csv(filename, sep="\s+", header=None, names=features, skiprows=1)
    return df

def load_biodarw(index, abs_paths):
    dataset = None
    for i, filename in zip(index, abs_paths) :
        tmp_df = read(filename)
        tmp_df['subject_id'] = i
        dataset = pd.concat([tmp_df,dataset])
    return dataset