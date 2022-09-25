import argparse
import sys


# data and processing
import numpy as np
import pandas as pd
from lstm import compile_and_fit, get_lstm_model
from datetime import date
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# ml
import tensorflow as tf
from sklearn.model_selection import train_test_split

# mlops
import mlflow
from datetime import date


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
parser.add_argument("--day", default=date.today().strftime("%Y%m%d"), type=str, help="%Y%m%d string where the files to be loaded where procesed ")


parser.add_argument("--labels_table", default="", type=str, help="labels table name in sql server")
parser.add_argument("--features_table", default="", type=str, help="features table name in sql server")

parser.add_argument("--db_user", default="airflow", type=str, help="database connection username")
parser.add_argument("--db_password", default="airflow", type=str, help="database connection password")
parser.add_argument("--db_host", default="192.168.1.154", type=str, help="database connection host")
parser.add_argument("--db_port", default="3307", type=str, help="database connection port")
parser.add_argument("--db_name", default="elekin", type=str, help="database name")

# Enable auto-logging to MLflow to capture TensorBoard metrics.
# export MLFLOW_TRACKING_URI=http://192.168.1.153:5001
# export GOOGLE_APPLICATION_CREDENTIALS: "/tmp/keys/eastonlab-b37c04a8f1a5.json"

def get_labels(labels_table ,engine):
    labels = pd.read_sql(labels_table ,engine, index_col='subject_id')
    return labels


def main(argv):

    args = parser.parse_args(argv[1:])

    if not tf.test.is_gpu_available():
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(tf.config.list_physical_devices('GPU')))

    mini_batch_size = args.mini_batch_size
    features_table = args.features_table 
    labels_table = args.labels_table

        # PyMySQL
    engine = create_engine(f'mysql+pymysql://{args.db_user}:{args.db_password}@{args.db_host}:{args.db_port}/{args.db_name}')
    df = pd.read_sql(features_table ,engine)
    labels = get_labels(labels_table, engine)

    if "window" not in features_table:
        df = df.T.sort_index()
    else:
        df.set_index("subject_id", inplace=True)
        df = df.sort_index()
    if "BIODARW" not in labels_table:
        labels.columns = ['labels']
        labels = df.join(labels).labels

    print(labels.value_counts())
    if "level" in labels_table:
        counts =  labels.value_counts()
        print(counts)
        ax = counts.plot.bar(title='Levels', x='counts')
        ax.bar_label(ax.containers[0])
        lb = LabelBinarizer()
        y = lb.fit_transform(labels.values.ravel()).astype(np.int16)
    else:
        counts = labels.value_counts()
        ax = counts.plot.bar(title='Control Subjects Vs. ET Subjects', x='counts')
        ax.bar_label(ax.containers[0])
        le = LabelEncoder().fit(labels.values.ravel())
        y = le.fit_transform((labels == 'si').astype(np.int16).values.ravel()).astype(np.int16)
    print(y.shape)
    
    X=df.values.astype(np.float32)
    print(X.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.test_ratio, random_state=args.seed)
    print(x_train.shape, x_test.shape)
    
    n_timesteps = x_train.shape[1]

    scaler = MinMaxScaler()
    scaler.fit(X)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).take(len(x_train)).batch(mini_batch_size).prefetch(AUTOTUNE).cache()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).take(len(x_test)).batch(mini_batch_size).prefetch(AUTOTUNE).cache()
    steps_per_epoch = round(len(train_dataset)/mini_batch_size)
    
    print("{0} train batches and {1} test batches of {2} mini batch size and {3} steps per epoch".format(len(train_dataset), 
                                                                            len(test_dataset),
                                                                            mini_batch_size,
                                                                            steps_per_epoch))

    

    clf = get_lstm_model(n_timesteps, 1 if not "level" in labels_table else 3, args.n_units, args.n_layers, drop_out=args.drop_out)
    
    loss = "binary_crossentropy" if not "level" in labels_table else "categorical_crossentropy"


    with mlflow.start_run(run_name=args.run_name):

        mlflow.tensorflow.autolog(every_n_iter=1)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("drop_out", args.drop_out)
        mlflow.log_param("mini_batch_size", mini_batch_size)
        mlflow.log_param("lstm_units", args.n_units)
        mlflow.log_param("n_outputs", args.n_classes)
        mlflow.set_tag("model", 'fcnn')
        mlflow.set_tag("class", labels_table[:5])
        mlflow.set_tag("features", features_table)

        compile_and_fit(clf, train_dataset, test_dataset,
                                seed=args.seed,
                                optimizer = tf.keras.optimizers.SGD(learning_rate=8e-4, momentum=0.9),
                                max_epochs=args.max_epoch, p_loss=loss)

if __name__ == "__main__":
    main(sys.argv)
