import tensorflow as tf
import tensorflow_addons as tfa
from typing import Tuple, List


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.models.Model):
    def __init__(self,
                 dim: int,
                 input_shape: Tuple,
                 kl_weight: float,
                 encoder_backbone: tf.keras.Model,
                 decoder_backbone: tf.keras.Model,
                 normalize: bool = False,
                 **kwargs):
        self.dim = dim
        self.input_shape_ = input_shape
        self.normalize = normalize
        self.kl_weight = tf.Variable(kl_weight, dtype=float, trainable=False)
        super(VAE, self).__init__(**kwargs)
        self.encoder_backbone = encoder_backbone
        self.decoder_backbone = decoder_backbone
        self.encoder = self.set_encoder()
        self.decoder = self.set_decoder()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.kl_weight_tracker = tf.keras.metrics.Mean(name="kl_weight")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def set_encoder(self) -> tf.keras.models.Model:
        input_encoder = tf.keras.layers.Input(self.input_shape_)
        x = self.encoder_backbone(input_encoder)
        if self.normalize:
            z_mean = tf.keras.layers.Dense(self.dim, name="z_mean")(x)
            z_mean = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(z_mean)
        else:
            z_mean = tf.keras.layers.Dense(self.dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.dim,
                                          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                          bias_initializer=tf.keras.initializers.Zeros(),
                                          name="z_log_var")(x)

        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.models.Model(input_encoder, [z_mean, z_log_var, z])
        return encoder

    def encode_images(self, input_images) -> List:
        """
        Takes array of images (n_images, *image_shape) and encodes them into the location parameter
        Args:
            input_images: array of shape (n_images, *image_shape)

        Returns:
            List of encodings with the embeddings per latent space
        """
        encodings = []
        # Iterate over all the latent spaces
        encodings.append(self.encoder.predict(input_images)[0])
        return encodings

    def set_decoder(self) -> tf.keras.models.Model:
        input_decoder = tf.keras.layers.Input((self.dim,))
        x = tf.keras.layers.Dense(self.decoder_backbone.inputs[0].shape[-1])(input_decoder)
        decoder = tf.keras.models.Model(input_decoder, self.decoder_backbone(x))
        return decoder

    def set_autoencoder(self):
        self.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        return self

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_weight_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print("Reconstruction shape", reconstruction.shape)
            print("Data shape", data.shape)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(-1, -2, -3)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) * self.kl_weight
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.kl_weight_tracker.update_state(self.kl_weight)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight_tracker.result()
        }

    def data_transformer(self, dictionary_):
        return dictionary_["image"][0]


class VAEClass(VAE):
    def __init__(self, dim, input_shape, kl_weight, encoder_backbone, decoder_backbone, num_labels, class_alpha=100,
                 normalize=False, **kwargs):
        self.num_labels = num_labels
        self.class_alpha = class_alpha
        super().__init__(dim, input_shape, kl_weight, encoder_backbone, decoder_backbone, normalize, **kwargs)
        self.classifier = self.set_classifier()
        self.classification_loss_tracker = tf.keras.metrics.Mean(name="class_loss")

    def set_classifier(self) -> tf.keras.models.Model:
        input_layer_classifier = tf.keras.layers.Input((self.dim,))
        x = tf.keras.layers.Dense(self.num_labels, activation="softmax")(input_layer_classifier)
        return tf.keras.models.Model(input_layer_classifier, x)

    def set_autoencoder(self):
        self.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        return self

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_weight_tracker,
            self.classification_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            image_input, (image_output, label) = data
            z_mean, z_log_var, z = self.encoder(image_input)
            reconstruction = self.decoder(z)
            class_prediction = self.classifier(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(image_input, reconstruction), axis=(1, 2)
                )
            )
            class_loss = self.class_alpha * tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(label, class_prediction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) * self.kl_weight
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss + class_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.classification_loss_tracker.update_state(class_loss)
        self.kl_weight_tracker.update_state(self.kl_weight)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.classification_loss_tracker.result(),
            "kl_weight": self.kl_weight_tracker.result()
        }

    def data_transformer(self, dictionary_):
        return dictionary_["image"][0], tf.one_hot(dictionary_["label"][0], depth=self.num_labels)


class VAETL(VAE):
    def __init__(self, dim, input_shape, kl_weight, encoder_backbone, decoder_backbone, class_alpha=1, **kwargs):
        self.class_alpha = class_alpha
        super().__init__(dim, input_shape, kl_weight, encoder_backbone, decoder_backbone, **kwargs)
        self.triplet_loss_tracker = tf.keras.metrics.Mean(name="triplet_loss")

    def set_autoencoder(self):
        self.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        return self

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_weight_tracker,
            self.triplet_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            image_input, (image_output, label) = data
            z_mean, z_log_var, z = self.encoder(image_input)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(image_input, reconstruction), axis=(1, 2)
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) * self.kl_weight
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            print("Kl loss", kl_loss)
            triplet_loss = tf.reduce_mean(tfa.losses.TripletSemiHardLoss()(label, z)) * self.class_alpha
            print("triplet loss", triplet_loss)
            total_loss = reconstruction_loss + kl_loss + triplet_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.kl_weight_tracker.update_state(self.kl_weight)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "triplet_loss": self.triplet_loss_tracker.result(),
            "kl_weight": self.kl_weight_tracker.result()
        }
