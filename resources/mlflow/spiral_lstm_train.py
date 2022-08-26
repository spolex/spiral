import argparse
import sys


# data and processing
import numpy as np
from utils import load_raw_data, compile_and_fit, get_model, get_optimizer


# ml
import tensorflow as tf


# mlops
import mlflow
import mlflow.tensorflow

# variables
FEATURES = ['x', 'y', 'pen_up', 'pressure']
COLS = [0, 1, 3, 6]


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="seed for random")
parser.add_argument("--n_classes", default=3, type=int, help="number of classes")
parser.add_argument("--mini_batch_size", default=4, type=int, help="size of mini batch")
parser.add_argument("--doc_path", default="/data/elekin/doc", type=str, help="metadata root directory")
parser.add_argument("--metadata_file", default="metadata-202106-v1.csv", type=str,
                    help="metadata file containing list of files to be loaded")
parser.add_argument("--test_ratio", default=0.33, type=float, help="train and test split ratio")
parser.add_argument("--run_name", default="lstm/tiny", type=str, help="name of run in MLOps system; mlflow")
parser.add_argument("--tracking_uri", default="http://192.168.1.154:5001", type=str,
                    help="URL of MLOps tracking system; mlflow")
parser.add_argument("--drop_out", default=0.5, type=float, help="seed for random")
parser.add_argument("--n_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--n_units", default=32, type=int, help="number of units for hidden layers")
parser.add_argument("--max_epoch", default=500, type=int, help="number of maximum epochs for training")

# Enable auto-logging to MLflow to capture TensorBoard metrics.
# export MLFLOW_TRACKING_URI=http://192.168.1.153:5001
# export GOOGLE_APPLICATION_CREDENTIALS: "/tmp/keys/eastonlab-b37c04a8f1a5.json"

def main(argv):
    args = parser.parse_args(argv[1:])
    mlflow.tensorflow.autolog(every_n_iter=1)
    with mlflow.start_run(run_name=args.run_name):
        if not tf.test.is_gpu_available():
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(tf.config.list_physical_devices('GPU')))

        seed = args.seed
        n_outputs = args.n_classes
        mini_batch_size = args.mini_batch_size

        x_train, y_train = load_raw_data(args.doc_path, args.metadata_file, FEATURES, COLS)
        print(np.array(x_train).shape)

        n_timesteps = np.array(x_train).shape[1]
        n_features = np.array(x_train).shape[2]

        data_size = np.array(x_train).shape[0]
        shuffle_buffer = data_size
        steps_per_epoch = round(data_size / mini_batch_size)

        AUTOTUNE = tf.data.experimental.AUTOTUNE

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
