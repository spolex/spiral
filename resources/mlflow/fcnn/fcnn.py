from random import seed
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

main_units = 64
secondary_units = 16
last_units = 8

def get_tiny_model(inputs, n_outputs, drop_out=.0, seed=38):    
    dense_1 = tf.keras.layers.Dense(main_units, activation=tf.nn.relu)(inputs)
    dropout_1 = tf.keras.layers.Dropout(drop_out, seed=seed,  )(dense_1)
    outputs = tf.keras.layers.Dense(n_outputs, activation=tf.nn.softmax if n_outputs > 1 else tf.nn.sigmoid)(dropout_1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_small_model(inputs, n_outputs,  drop_out=.0, seed=38):    
    dense_1 = tf.keras.layers.Dense(main_units, activation=tf.nn.relu)(inputs)
    dropout_1 = tf.keras.layers.Dropout(drop_out, seed=seed)(dense_1)
    dense_2 = tf.keras.layers.Dense(last_units, activation=tf.nn.relu)(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(drop_out, seed=seed)(dense_2)
    outputs = tf.keras.layers.Dense(n_outputs, activation=tf.nn.softmax if n_outputs > 1 else tf.nn.sigmoid)(dropout_2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_large_model(inputs, n_outputs, drop_out=.0, seed=38):    
    dense_1 = tf.keras.layers.Dense(main_units, activation=tf.nn.relu)(inputs)
    dropout_1 = tf.keras.layers.Dropout(drop_out, seed=seed)(dense_1)
    dense_2 = tf.keras.layers.Dense(secondary_units, activation=tf.nn.relu)(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(drop_out, seed=seed)(dense_2)
    dense_3 = tf.keras.layers.Dense(secondary_units, activation=tf.nn.relu)(dropout_2)
    dropout_3 = tf.keras.layers.Dropout(drop_out, seed=seed)(dense_3)
    dense_4 = tf.keras.layers.Dense(last_units, activation=tf.nn.relu)(dropout_3)
    dropout_5 = tf.keras.layers.Dropout(drop_out, seed=seed)(dense_4)
    outputs = tf.keras.layers.Dense(n_outputs, activation=tf.nn.softmax if n_outputs > 1 else tf.nn.sigmoid)(dropout_5)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

## Early stop configuration
def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=100, min_delta=1e-4, mode='min'),
    ]

def compile_and_fit(model, train_dataset, test_dataset, seed, optimizer=None, max_epochs=1e3, 
                    p_loss='categorical_crossentropy',
                    p_metrics=['accuracy','Precision', 'Recall', 'TruePositives', 'FalsePositives', 'TrueNegatives', 'FalseNegatives']):
    """"""
    tf.keras.backend.clear_session()  # avoid clutter from old models and layers, especially when memory is limited
    
    model.compile(optimizer=optimizer, loss=p_loss, metrics=p_metrics)
    model.summary()
    
    tf.random.set_seed(seed)  # establecemos la semilla para tensorflow
    
    history = model.fit(train_dataset, validation_data=test_dataset,
                        use_multiprocessing=True, verbose=1, shuffle=False,
                        epochs=max_epochs, callbacks=get_callbacks())
    return history
