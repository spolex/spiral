import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib


logdir = pathlib.Path("C:/tmp/")/"tensorboard_logs"
#logdir = pathlib.Path("gdrive/My Drive/Colab Notebooks/elekin")/"tensorboard_logs"


"""Training helpper functions
"""
def get_callbacks(name, reports_every=10, patience=100):
    return [
        tfdocs.modeling.EpochDots(report_every=reports_every),
        #tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=100),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience),
        tf.keras.callbacks.TensorBoard(logdir/name)
      ]

def get_callbacks_t(name, reports_every=10, patience=200):
    return [
        tfdocs.modeling.EpochDots(report_every=reports_every),
        tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=patience),
        tf.keras.callbacks.TensorBoard(logdir/name)
      ]

"""Plot accuracy for train and val datasets and also loss function vs number of epoch
"""
def plot_report(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
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
def build_basic_lstm(num_cells, size, normalize=True, dropout=0.15): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=(size,)))
    model.add(tf.keras.layers.LSTM(num_cells, activation=tf.nn.tanh, return_sequences=False))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
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
    ds = ds.map(lambda window: (window[:-1], label))
    #ds = ds.batch(batch_size).prefetch(1) # better batch after concatenate and shuffle all the windowed samples
    return ds


def split_train_test_val(full_dataset, data_size, train_ratio, val_ratio, test_ratio, shuffle_buffer=16e6):
    """Create dataset splits for training pruposes from tensorflow dataset full_dataset
    """
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    test_size = int(test_ratio * data_size)

    full_dataset = full_dataset.shuffle(shuffle_buffer)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    return (train_dataset, val_dataset, test_dataset)