
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

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
    
    model.compile(optimizer=get_optimizer() if optimizer is None else optimizer, loss=p_loss, metrics=p_metrics)
    model.summary()
    
    tf.random.set_seed(seed)  # establecemos la semilla para tensorflow
    
    history = model.fit(train_dataset, validation_data=test_dataset,
                        use_multiprocessing=True, verbose=0, shuffle=False,
                        epochs=max_epochs, callbacks=get_callbacks())
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