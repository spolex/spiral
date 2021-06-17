import argparse
from os import path
import sys
import logging

# data and processing
import pandas as pd
import numpy as np
from scipy.signal import resample

# ml
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import tensorflow_docs as tfdocs
from tensorflow_docs import modeling


# mlops
import mlflow
import mlflow.tensorflow

FEATURES = ['x', 'y', 'pen_up', 'pressure']


def load_data(doc_path, filename):
    """ """
    meta_df = pd.read_csv(path.join(doc_path, filename), index_col=0)
    x_train = []
    y_train = []

    for file_path, level in zip(meta_df.abs_path, meta_df.level):
        df = pd.read_csv(file_path, sep="\s+", header=None, names=FEATURES, skiprows=1, usecols=[0, 1, 3, 6])
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
        tfdocs.modeling.EpochDots(),
        earlystop_callback,
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2e2, min_delta=1e-5),
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
        model.add(tf.keras.layers.Dropout(drop_out))
        model.add(tf.keras.layers.LSTM(n_units, activation=tf.nn.tanh, return_sequences=False,
                                       name='lstm_hidden_layer'))

    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, name='dense_hidden_layer'))
    model.add(tf.keras.layers.Dense(n_outputs, activation=tf.nn.sigmoid, name='output'))
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=38, type=int, help="seed for random")
parser.add_argument("--n_classes", default=3, type=int, help="number of classes")
parser.add_argument("--mini_batch_size", default=4, type=int, help="size of mini batch")
parser.add_argument("--doc_path", default="/data/elekin/doc", type=str, help="metadata root directory")
parser.add_argument("--metadata_file", default="metadata-202106-v1.csv", type=str,
                    help="metadata file containing list of files to be loaded")
parser.add_argument("--test_ratio", default=0.33, type=float, help="train and test split ratio")
parser.add_argument("--run_name", default="lstm/tiny", type=str, help="name of run in MLOps system; mlflow")
parser.add_argument("--tracking_uri", default="http://192.168.1.12:5001", type=str,
                    help="URL of MLOps tracking system; mlflow")
parser.add_argument("--drop_out", default=0.5, type=float, help="seed for random")
parser.add_argument("--n_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--n_units", default=32, type=int, help="number of units for hidden layers")
parser.add_argument("--max_epoch", default=500, type=int, help="number of maximum epochs for training")

# Enable auto-logging to MLflow to capture TensorBoard metrics.
# export MLFLOW_TRACKING_URI=http://192.168.1.12:5001
# export GOOGLE_APPLICATION_CREDENTIALS: "/tmp/keys/eastonlab-b37c04a8f1a5.json"

def main(argv):
    args = parser.parse_args(argv[1:])
    # mlflow.set_tracking_uri("http://192.168.1.12:5001")
    # mlflow.set_experiment('/archimedes-dl')
    mlflow.tensorflow.autolog()
    with mlflow.start_run(run_name=args.run_name):
        if not tf.config.list_physical_devices('GPU'):
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(tf.config.list_physical_devices('GPU')))

        # mlflow.set_tracking_uri(args.tracking_uri)
        # mlflow.set_experiment('/archimedes-dl')

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        seed = args.seed
        n_outputs = args.n_classes
        mini_batch_size = args.mini_batch_size

        x_train, y_train = load_data(args.doc_path, args.metadata_file)
        n_timesteps = np.array(x_train).shape[1]
        n_features = np.array(x_train).shape[2]
        data_size = np.array(x_train).shape[0]
        shuffle_buffer = data_size
        steps_per_epoch = round(data_size / mini_batch_size)

        train_split = 1.0 - args.test_ratio
        test_split = args.test_ratio
        train_size = int(train_split * data_size)
        # test_size = int(test_split * data_size)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, tf.one_hot(y_train, n_outputs)))
        full_dataset = dataset.shuffle(shuffle_buffer, seed=seed)
        train_dataset = full_dataset.take(train_size).batch(mini_batch_size).prefetch(AUTOTUNE).cache()
        test_dataset = full_dataset.skip(train_size).batch(mini_batch_size).prefetch(AUTOTUNE).cache()

        mlflow.log_param("seed", seed)
        mlflow.log_param("drop_out", args.drop_out)
        mlflow.log_param("mini_batch_size", mini_batch_size)
        mlflow.log_param("train_split", train_split)
        mlflow.log_param("test_split", test_split)
        mlflow.log_param("lstm_units", args.n_units)
        mlflow.log_param("features", FEATURES)

        clf = get_model(n_features, n_timesteps, n_outputs, args.n_units, args.n_layers, args.drop_out)
        history = compile_and_fit(clf, train_dataset, test_dataset,
                                  seed=args.seed,
                                  optimizer=get_optimizer(steps_per_epoch, 1e-3),
                                  max_epochs=args.max_epoch)

        print("\n#######################Evaluation###########################")
        # Evaluate the model on the test data using `evaluate`
        print('train acc:', max(history.history["accuracy"]))
        print('test acc:', max(history.history["val_accuracy"]))


if __name__ == "__main__":
    main(sys.argv)
