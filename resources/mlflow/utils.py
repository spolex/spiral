from os import path
from scipy.signal import resample
from pandas import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


def load_raw_data(doc_path, filename, features, cols):
    """ """
    meta_df = pd.read_csv(path.join(doc_path, filename), index_col=0)
    x_train = []
    y_train = []

    for file_path, level in zip(meta_df.abs_path, meta_df.level):
        df = pd.read_csv(file_path, sep="\s+", header=None, names=features, skiprows=1, usecols=cols)
        x_train.append(resample(df.values.astype('int16'), 4096))
        y_train.append(level)
    return x_train, y_train

# #Early stop configuration
earlystop_callback = EarlyStopping(
  monitor='val_accuracy', min_delta=1e-3,
  patience=200)

training_earlystop_callback = EarlyStopping(
    monitor='accuracy', min_delta=1e-4,
    patience=200)


def get_callbacks():
    return [
        #tfdocs.modeling.EpochDots(),
        earlystop_callback,
        EarlyStopping(monitor='val_loss', patience=2e2, min_delta=1e-5),
    ]

def compile_and_fit(model, train_dataset, test_dataset, seed, optimizer=None, max_epochs=1e3):
    tf.keras.backend.clear_session()  # avoid clutter from old models and layers, especially when memory is limited
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    tf.random.set_seed(seed)  # establecemos la semilla para tensorflow
    history = model.fit(train_dataset,
                        use_multiprocessing=True,
                        validation_data=test_dataset, epochs=max_epochs,
                        callbacks=get_callbacks(),
                        verbose=1, shuffle=True)
    return history

# Many models train better if you gradually reduce the learning rate during training.
# Use optimizers.schedules to reduce the learning rate over time:
def get_optimizer(steps_per_epoch=1, lr=1e-4, multiplier=1e3):
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(lr,
                                                                 decay_steps=steps_per_epoch * multiplier,
                                                                 decay_rate=1,
                                                                 staircase=False)
    return tf.keras.optimizers.Adam(lr_schedule)


def get_model(n_features, n_timesteps, n_outputs, n_units, n_layers=1, drop_out=0.5):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(n_units, activation=tf.nn.tanh, return_sequences=n_layers > 1,
                                   input_shape=(n_timesteps, n_features)))

    for n_layer in range(1, n_layers):
        model.add(tf.keras.layers.LSTM(n_units, activation=tf.nn.tanh, return_sequences=n_layer!=n_layers-1,
                                       name='lstm_hidden_layer_{}'.format(n_layer)))
        model.add(tf.keras.layers.Dropout(drop_out))

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, name='dense_hidden_layer'))
    model.add(tf.keras.layers.Dense(n_outputs, activation=tf.nn.sigmoid, name='output'))
    return model