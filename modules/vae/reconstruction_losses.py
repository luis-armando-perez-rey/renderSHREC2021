from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf


def gaussian_loss(dec_std=1/(2**0.5)):
    dec_std = float(dec_std)  # cannot take K.log of an int

    def loss(x_in, x_out):
        return K.sum(K.square(K.batch_flatten(x_in) - K.batch_flatten(x_out)) / (2 * dec_std**2) +
                     K.log(dec_std) + 0.5 * K.log(2 * np.pi), axis=-1)
    return loss


def bernoulli_loss():
    def loss(x_in, x_out):
        data_dim = tf.reduce_prod(tf.cast(x_in.shape[1:], tf.float32))
        return data_dim * binary_crossentropy(K.batch_flatten(x_in), K.batch_flatten(x_out))
    return loss
