import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_autoencoder(input_dim:int):
    inp=tf.keras.Input(shape=(input_dim,))
    x=layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-4))(inp)
    x=layers.Dropout(0.2)(x)
    x=layers.Dense(64,activation='relu')(x)
    bottleneck=layers.Dense(32,activation='relu')(x)
    x=layers.Dense(64,activation='relu')(bottleneck)
    x=layers.Dense(128,activation='relu')(x)
    out=layers.Dense(input_dim,activation='sigmoid')(x)
    model=tf.keras.Model(inp,out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='mse')
    return model
