import argparse
from cgitb import small
import sys


# data and processing
import numpy as np
import pandas as pd
from fcnn import get_tiny_model, get_small_model, get_large_model, compile_and_fit

# ml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# mlops
import mlflow
from datetime import date


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="seed for random")
parser.add_argument("--n_classes", default=3, type=int, help="number of classes")
parser.add_argument("--mini_batch_size", default=4, type=int, help="size of mini batch")
parser.add_argument("--doc_path", default="/data/elekin/doc", type=str, help="metadata root directory")
parser.add_argument("--metadata_file", default="metadata-202208-v1.csv", type=str, help="metadata file containing list of files to be loaded")
parser.add_argument("--test_ratio", default=0.33, type=float, help="train and test split ratio")
parser.add_argument("--drop_out", default=0.5, type=float, help="")
parser.add_argument("--model", default="tiny", type=str, help="name of run in MLOps system; mlflow")
parser.add_argument("--tracking_uri", default="http://192.168.1.154:5001", type=str, help="URL of MLOps tracking system; mlflow")
parser.add_argument("--max_epoch", default=500, type=int, help="number of maximum epochs for training")
parser.add_argument("--day", default=date.today().strftime("%Y%m%d"), type=str, help="%Y%m%d string where the files to be loaded where procesed ")
parser.add_argument("--filepath", default="/data/elekin/data/results/handwriting/residues_17_20220911.csv", type=str, help="location of the file with dataset for training")


# Enable auto-logging to MLflow to capture TensorBoard metrics.
# export MLFLOW_TRACKING_URI=http://192.168.1.153:5001
# export GOOGLE_APPLICATION_CREDENTIALS: "/tmp/keys/eastonlab-b37c04a8f1a5.json"

def main(argv):

    args = parser.parse_args(argv[1:])
        
    if not tf.test.is_gpu_available():
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(tf.config.list_physical_devices('GPU')))

    seed = args.seed
    mini_batch_size = args.mini_batch_size

    df = pd.read_csv(args.filepath, index_col=0)
    if "window" not in args.filepath:
        df = df.T.sort_index()
    else:
        df = df.sort_index()

    labels = None
    y = None
    ax = None
    if args.n_classes > 1:
        labels = pd.read_csv("/data/elekin/data/results/handwriting/level_20220903.csv", index_col=0).sort_index()
        labels.columns = ['labels']
        labels = df.join(labels).labels
        counts =  labels.value_counts()
        print(counts)
        ax =counts.plot.bar(title='Levels', x='counts')
        ax.bar_label(ax.containers[0])
        lb = preprocessing.LabelBinarizer()
        y = lb.fit_transform(labels.values.ravel()).astype(np.int16)
    else:
        labels = pd.read_csv("/data/elekin/data/results/handwriting/binary_labels_20220903.csv", index_col=0).sort_index()
        labels.columns = ['labels']
        labels = df.join(labels).labels
        le = preprocessing.LabelEncoder().fit(labels.values.ravel())
        y = le.fit_transform((labels == 'si').astype(np.int16).values.ravel()).astype(np.int16)
        counts =  labels.value_counts()
        ax =counts.plot.bar(title='Control Subjects Vs. ET Subjects', x='counts')
        ax.bar_label(ax.containers[0])

    print(y.shape)
    X=df.values.astype(np.float32)
    print(X.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.test_ratio, random_state=args.seed)
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    print(x_train.shape, x_test.shape)
    n_features = x_train.shape[1]

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.from_tensor_slices((scaler.transform(x_train), y_train)).take(len(x_train)).batch(mini_batch_size).prefetch(AUTOTUNE).cache()
    test_dataset = tf.data.Dataset.from_tensor_slices((scaler.transform(x_test), y_test)).take(len(x_test)).batch(mini_batch_size).prefetch(AUTOTUNE).cache()
    steps_per_epoch = round(len(train_dataset)/mini_batch_size)
    
    print("{0} train batches and {1} test batches of {2} mini batch size and {3} steps per epoch".format(len(train_dataset), 
                                                                            len(test_dataset),
                                                                            mini_batch_size,
                                                                            steps_per_epoch))

    inputs = tf.keras.layers.Input(shape=(n_features,))

    models = { "tiny" : get_tiny_model(inputs=inputs, n_outputs=args.n_classes, drop_out=args.drop_out),
                "small" : get_small_model(inputs=inputs, n_outputs=args.n_classes, drop_out=args.drop_out),
                "large" : get_large_model(inputs=inputs, n_outputs=args.n_classes, drop_out=args.drop_out)}


    loss = "binary_crossentropy" if args.n_classes < 2 else "categorical_crossentropy"

    
    with mlflow.start_run(run_name=args.model):
        mlflow.log_figure(ax.figure, "labels.png")
        mlflow.tensorflow.autolog(every_n_iter=1)

        compile_and_fit(models[args.model], train_dataset, test_dataset,
                                seed=args.seed,
                                optimizer=tf.keras.optimizers.Adam(1e-4),
                                max_epochs=args.max_epoch, p_loss=loss)

        mlflow.end_run()
            

if __name__ == "__main__":
    main(sys.argv)
