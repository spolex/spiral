from datetime import date
from os import path
from scipy.signal import resample
from pandas import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
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

## Early stop configuration
def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=100, min_delta=1e-4, mode='min'),
    ]

def compile_and_fit(model, train_dataset, test_dataset, seed, optimizer=None, max_epochs=1e3, p_metrics=['accuracy','Precision', 'Recall', 'TruePositives', 'FalsePositives', 'TrueNegatives', 'FalseNegatives']):
    tf.keras.backend.clear_session()  # avoid clutter from old models and layers, especially when memory is limited
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=p_metrics)
    model.summary()
    tf.random.set_seed(seed)  # establecemos la semilla para tensorflow
    history = model.fit(train_dataset,
                        use_multiprocessing=True,
                        validation_data=test_dataset, epochs=max_epochs,
                        callbacks=get_callbacks(),
                        verbose=1)
    return history

def binary_compile_and_fit(model, train_dataset, test_dataset, seed, optimizer=None, max_epochs=1e3, p_metrics=['accuracy','Precision', 'Recall', 'TruePositives', 'FalsePositives', 'TrueNegatives', 'FalseNegatives']):
    tf.keras.backend.clear_session()  # avoid clutter from old models and layers, especially when memory is limited
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=p_metrics)
    model.summary()
    tf.random.set_seed(seed)  # establecemos la semilla para tensorflow
    history = model.fit(train_dataset,
                        use_multiprocessing=True,
                        validation_data=test_dataset, epochs=max_epochs,
                        callbacks=get_callbacks(),
                        verbose=1)
    return history

# Many models train better if you gradually reduce the learning rate during training.
# Use optimizers.schedules to reduce the learning rate over time:
def get_optimizer(steps_per_epoch=1, lr=1e-4, multiplier=1e3):
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(lr,
                                                                 decay_steps=steps_per_epoch * multiplier,
                                                                 decay_rate=1,
                                                                 staircase=False)
    return tf.keras.optimizers.Adam(lr_schedule)
        

def get_lstm_model(n_timesteps, n_outputs, n_units, n_layers=1, drop_out=0.5, fcnn_units=8):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),# expand the dimension form (50, 4096) to (50, 4096, 1)
                      input_shape=[n_timesteps,]))
    model.add(tf.keras.layers.LSTM(n_units, activation=tf.nn.tanh, return_sequences=n_layers > 1))
    model.add(tf.keras.layers.Dropout(drop_out))

    for n_layer in range(1, n_layers):
        model.add(tf.keras.layers.LSTM(n_units, activation=tf.nn.tanh, return_sequences=n_layer!=n_layers-1,
                                       name='lstm_hidden_layer_{}'.format(n_layer)))
        model.add(tf.keras.layers.Dropout(drop_out))

    model.add(tf.keras.layers.Dense(fcnn_units, activation=tf.nn.relu, name='dense_hidden_layer'))
    model.add(tf.keras.layers.Dense(n_outputs, activation=tf.nn.softmax if n_outputs > 1 else tf.nn.sigmoid, name='output'))
    return model

    

