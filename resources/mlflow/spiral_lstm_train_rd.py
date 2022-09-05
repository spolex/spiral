import argparse
import sys


# data and processing
import numpy as np
import pandas as pd
from utils import compile_and_fit, get_model_rd, get_optimizer


# ml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# mlops
import mlflow


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="seed for random")
parser.add_argument("--n_classes", default=3, type=int, help="number of classes")
parser.add_argument("--mini_batch_size", default=4, type=int, help="size of mini batch")
parser.add_argument("--doc_path", default="/data/elekin/doc", type=str, help="metadata root directory")
parser.add_argument("--metadata_file", default="metadata-202208-v1.csv", type=str,
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

        residues = pd.read_csv("/data/elekin/data/results/handwriting/residues_17_20220901.csv")
        residues = residues.set_index(residues.columns[0]).sort_index()
        X=residues.values.astype(np.float64)
        print(X.shape)

        labels = pd.read_csv("/data/elekin/data/results/handwriting/level_20220901.csv", index_col=0).sort_index()
        lb = preprocessing.LabelBinarizer()
        y = lb.fit_transform(labels).astype(np.int16)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=38)
        print(x_train.shape, y_train.shape)

        n_features = x_train.shape[1]

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).take(len(x_train)).batch(mini_batch_size).prefetch(AUTOTUNE).cache()
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).take(len(x_test)).batch(mini_batch_size).prefetch(AUTOTUNE).cache()
        steps_per_epoch = round(len(train_dataset)/mini_batch_size)
        print("{0} train batches and {1} test batches of {2} mini batch size and {3} steps per epoch".format(len(train_dataset), 
                                                                              len(test_dataset),
                                                                              mini_batch_size,
                                                                                steps_per_epoch))

        mlflow.log_param("seed", seed)
        mlflow.log_param("drop_out", args.drop_out)
        mlflow.log_param("mini_batch_size", mini_batch_size)
        mlflow.log_param("lstm_units", args.n_units)
        mlflow.log_param("n_outputs", args.n_classes)

        clf = get_model_rd(n_features, n_outputs, args.n_units, args.n_layers, drop_out=args.drop_out)
    
        history = compile_and_fit(clf, train_dataset, test_dataset,
                                  seed=args.seed,
                                  optimizer=tf.keras.optimizers.Adam(1e-4),
                                  max_epochs=args.max_epoch)

        print("\n#######################Evaluation###########################")
        # Evaluate the model on the test data using `evaluate`
        print('train acc:', max(history.history["accuracy"]))
        print('test acc:', max(history.history["val_accuracy"]))


if __name__ == "__main__":
    main(sys.argv)