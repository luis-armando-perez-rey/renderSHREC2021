import tensorflow as tf
import tensorflow_addons as tfa


class AE(tf.keras.models.Model):
    def __init__(self, dim, input_shape, encoder_backbone, decoder_backbone, normalize=False, **kwargs):
        self.dim = dim
        self.input_shape_ = input_shape
        self.normalize = normalize
        super(AE, self).__init__(**kwargs)
        self.encoder_backbone = encoder_backbone
        self.decoder_backbone = decoder_backbone
        self.encoder = self.set_encoder()
        self.decoder = self.set_decoder()
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )

    def set_encoder(self) -> tf.keras.models.Model:
        input_encoder = tf.keras.layers.Input(self.input_shape_)
        x = self.encoder_backbone(input_encoder)
        if self.normalize:
            z_mean = tf.keras.layers.Dense(self.dim, name="z_mean")(x)
            z_mean = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(z_mean)
        else:
            z_mean = tf.keras.layers.Dense(self.dim, name="z_mean")(x)
        encoder = tf.keras.models.Model(input_encoder, z_mean)
        return encoder

    def set_decoder(self) -> tf.keras.models.Model:
        input_decoder = tf.keras.layers.Input((self.dim,))
        x = tf.keras.layers.Dense(self.decoder_backbone.inputs[0].shape[-1])(input_decoder)
        decoder = tf.keras.models.Model(input_decoder, self.decoder_backbone(x))
        return decoder

    def set_autoencoder(self):
        self.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        return self

    def call(self, inputs, training=False):
        z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            print("Reconstruction shape", reconstruction.shape)
            print("Data shape", data.shape)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(-1, -2, -3)
                )
            )

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

    def data_transformer(self, dictionary_):
        return dictionary_["image"][0]



class AEClass(AE):
    def __init__(self, dim, input_shape, encoder_backbone, decoder_backbone, num_labels, class_alpha=1.0,
                 normalize=False, **kwargs):
        super(AE, self).__init__(dim, input_shape, encoder_backbone, decoder_backbone, normalize=normalize, **kwargs)
        self.num_labels = num_labels
        self.class_alpha = class_alpha
        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def set_classifier(self) -> tf.keras.models.Model:
        input_layer_classifier = tf.keras.layers.Input((self.dim,))
        x = tf.keras.layers.Dense(self.num_labels, activation="softmax")(input_layer_classifier)
        return tf.keras.models.Model(input_layer_classifier, x)

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.class_loss_tracker,
            self.total_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            image_input, (image_output, label) = data
            z = self.encoder(image_input)

            class_prediction = self.classifier(z)

            reconstruction = self.decoder(z)
            print("Reconstruction shape", reconstruction.shape)
            print("Data shape", data.shape)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(-1, -2, -3)
                )
            )
            class_loss = self.class_alpha * tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(label, class_prediction))
            total_loss = reconstruction_loss + class_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
        }


    def data_transformer(self, dictionary_):
        return dictionary_["image"][0], tf.one_hot(dictionary_["label"][0], depth=self.num_classes)

class AETL(AE):
    def __init__(self, dim, input_shape, encoder_backbone, decoder_backbone, class_alpha=1.0, normalize=False,
                 **kwargs):
        super(AE, self).__init__(dim, input_shape, encoder_backbone, decoder_backbone, normalize, **kwargs)
        self.class_alpha = class_alpha
        self.tl_loss_tracker = tf.keras.metrics.Mean(name="tl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.tl_loss_tracker,
            self.total_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            image_input, (image_output, label) = data
            z = self.encoder(image_input)
            reconstruction = self.decoder(z)
            print("Reconstruction shape", reconstruction.shape)
            print("Data shape", data.shape)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(-1, -2, -3)
                )
            )
            triplet_loss = tf.reduce_mean(tfa.losses.TripletSemiHardLoss()(label, z)) * self.class_alpha
            total_loss = reconstruction_loss + triplet_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.tl_loss_tracker.update_state(triplet_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "tl_loss": self.tl_loss_tracker.result(),
        }

    def data_transformer(self, dictionary_):
        return dictionary_["image"][0], dictionary_["label"][0]
