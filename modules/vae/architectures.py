from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, \
    BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


def encoder_decoder_dense(input_shape=(28, 28, 1), activation="relu",
                          dense_units_lst=(512, 256)):
    # ENCODER
    x = Input(shape=input_shape)
    h = Flatten()(x)
    for units in dense_units_lst:
        h = Dense(units, activation=activation)(h)
        h = BatchNormalization()(h)
    encoder = Model(x, h)

    # DECODER
    dec_in = Input(shape=(dense_units_lst[-1],))
    h = dec_in
    for units in reversed(dense_units_lst[:-1]):
        h = Dense(units, activation=activation)(h)
        h = BatchNormalization()(h)
    h = Dense(np.prod(input_shape), activation="sigmoid")(h)
    x_reconstr = Reshape(input_shape)(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder


def encoder_decoder_vgglike_2d(height=224, width=224, depth=1, activation="relu",
                               filters_lst=(64, 64, 64),
                               kernel_size_lst=((3, 3), (3, 3), (3, 3)),
                               pool_size_lst=((2, 2), (2, 2), (2, 2)),
                               dense_units_lst=(64,),
                               batchnorm=False):
    assert len(filters_lst) == len(kernel_size_lst) == len(pool_size_lst), \
        "lists for filters/kernel_size/pool_size must be of the same length"
    n_conv_layers = len(filters_lst)
    input_shape = (height, width, depth)
    # calculate sizes after convolutions (before dense layers)
    conv_height = height
    conv_width = width
    for (vertical, horizontal) in pool_size_lst:
        conv_height = int(conv_height / vertical)
        conv_width = int(conv_width / horizontal)
    conv_depth = filters_lst[-1]
    conv_dim = conv_height * conv_width * conv_depth

    # ENCODER
    x = Input(shape=input_shape)
    # convolutional layers
    h = x
    for i in range(n_conv_layers):
        h = Conv2D(filters=filters_lst[i], kernel_size=kernel_size_lst[i], strides=(1, 1), padding="same",
                   activation=activation)(h)
        h = MaxPooling2D(pool_size=pool_size_lst[i], padding="same")(h)
        if batchnorm:
            h = BatchNormalization()(h)
    # dense layers
    h = Flatten()(h)
    for units in dense_units_lst:
        h = Dense(units, activation=activation)(h)
        if batchnorm:
            h = BatchNormalization()(h)
    encoder = Model(x, h)

    # DECODER
    dec_in = Input(shape=(dense_units_lst[-1],))
    h = dec_in
    # dense layers
    for units in reversed(dense_units_lst[:-1]):
        h = Dense(units, activation=activation)(h)
        if batchnorm:
            h = BatchNormalization()(h)
    h = Dense(conv_dim, activation=activation)(h)
    h = Reshape((conv_height, conv_width, conv_depth))(h)
    if batchnorm:
        h = BatchNormalization()(h)
    # convolutional layers
    for i in reversed(range(1, n_conv_layers)):
        h = UpSampling2D(size=pool_size_lst[i])(h)
        h = Conv2D(filters=filters_lst[i - 1], kernel_size=kernel_size_lst[i], strides=(1, 1), padding="same",
                   activation=activation)(h)
        if batchnorm:
            h = BatchNormalization()(h)
    h = UpSampling2D(size=pool_size_lst[0])(h)
    x_reconstr = Conv2D(filters=depth, kernel_size=kernel_size_lst[0], strides=(1, 1), padding="same",
                        activation="sigmoid")(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder


def encoder_decoder_dislib_2d(height=64, width=64, depth=1, activation="relu",
                              filters_lst=(32, 32, 64, 64),
                              kernel_size_lst=((4, 4), (4, 4), (4, 4), (4, 4)),
                              dense_units_lst=(256,),
                              ):
    assert len(filters_lst) == len(kernel_size_lst), \
        "lists for filters/kernel_size/pool_size must be of the same length"
    n_conv_layers = len(filters_lst)
    input_shape = (height, width, depth)

    # ENCODER
    x = Input(shape=input_shape)
    # convolutional layers
    h = x
    for i in range(n_conv_layers):
        h = Conv2D(filters=filters_lst[i], kernel_size=kernel_size_lst[i], strides=(2, 2), padding="same",
                   activation=activation)(h)
    # dense layers
    h = Flatten()(h)
    for units in dense_units_lst:
        h = Dense(units, activation=activation)(h)
    encoder = Model(x, h)

    # DECODER
    dec_in = Input(shape=(dense_units_lst[-1],))
    h = dec_in
    conv_dim = 4 * 4 * 64
    # dense layers
    for units in reversed(dense_units_lst[:-1]):
        h = Dense(units, activation=activation)(h)

    h = Dense(conv_dim, activation=activation)(h)
    h = Reshape((4, 4, 64))(h)
    # convolutional layers
    for i in reversed(range(0, n_conv_layers - 1)):
        h = Conv2DTranspose(filters=filters_lst[i], kernel_size=kernel_size_lst[i], strides=(2, 2), padding="same",
                            activation=activation)(h)

    x_reconstr = Conv2DTranspose(filters=depth, kernel_size=kernel_size_lst[0], strides=(2, 2), padding="same",
                                 activation="sigmoid")(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder




def get_encoder_decoder(architecture, image_shape):
    if architecture == "vgg":
        architecture_parameters = {"height": image_shape[0],
                                   "width": image_shape[1],
                                   "depth": image_shape[2],
                                   "filters_lst": (64, 128, 256, 512, 512),
                                   "kernel_size_lst": ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3)),
                                   "pool_size_lst": ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
                                   "dense_units_lst": (4096, 1000,)
                                   }
        architecture_function = encoder_decoder_vgglike_2d

    elif architecture == "dense":
        architecture_parameters = {"input_shape": image_shape,
                                   "dense_units_lst": (512, 512, 256, 100)}
        architecture_function = encoder_decoder_dense

    elif architecture == "resnet50v2_dense":

        architecture_parameters = {"encoder_params": {"include_top": False,
                                                      "weights": "imagenet",
                                                      "pooling": "avg",
                                                      },
                                   "decoder_params":
                                       {"input_shape": image_shape,
                                        "dense_units_lst": (512, 512, 256)}
                                   }

        def architecture_function(encoder_params, decoder_params):
            encoder_preload = tf.keras.applications.ResNet50V2(**encoder_params)
            encoder_preload.trainable = False
            input_layer = tf.keras.layers.Input(image_shape)
            x = encoder_preload(input_layer)
            x = tf.keras.layers.Dense(1000, activation="relu")(x)
            encoder = tf.keras.models.Model(input_layer, x)
            _, decoder = encoder_decoder_dense(**decoder_params)
            return encoder, decoder
    elif architecture == "resnet50v2_vgg":

        architecture_parameters = {"encoder_params": {"include_top": False,
                                                      "weights": "imagenet",
                                                      "pooling": "avg",
                                                      },
                                   "decoder_params": {"height": image_shape[0],
                                   "width": image_shape[1],
                                   "depth": image_shape[2],
                                   "filters_lst": (64, 128, 256, 512, 512),
                                   "kernel_size_lst": ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3)),
                                   "pool_size_lst": ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
                                   "dense_units_lst": (4096, 1000,)}
                                   }

        def architecture_function(encoder_params, decoder_params):
            encoder_preload = tf.keras.applications.ResNet50V2(**encoder_params)
            encoder_preload.trainable = False
            input_layer = tf.keras.layers.Input(image_shape)
            x = encoder_preload(input_layer)
            x = tf.keras.layers.Dense(1000, activation="relu")(x)
            encoder = tf.keras.models.Model(input_layer, x)
            _, decoder = encoder_decoder_vgglike_2d(**decoder_params)
            return encoder, decoder
    elif architecture == "dis_lib":
        architecture_parameters = {"height":image_shape[0],
                                   "width":image_shape[1],
                                   "depth":image_shape[2],
                                   "activation":"relu",
                                  "filters_lst":(32, 32, 64, 64),
                                  "kernel_size_lst":((4, 4), (4, 4), (4, 4), (4, 4)),
                                  "dense_units_lst":(256,),

        }
        architecture_function = encoder_decoder_dislib_2d
    else:
        architecture_parameters = None
        architecture_function = None

    encoder_backbone, decoder_backbone = architecture_function(**architecture_parameters)

    return encoder_backbone, decoder_backbone
