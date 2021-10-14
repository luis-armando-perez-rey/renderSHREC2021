from tensorflow.keras.layers import Concatenate, Dense, Input, TimeDistributed, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
from typing import List

import sys
import os
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_PATH)
from modules.latent_space.latentspace2 import LatentSpace


class TransformVAE(tf.keras.Model):
    """docstring for TransformVAE"""

    def __init__(self,
                 encoders: List[tf.keras.models.Model],
                 decoder: tf.keras.models.Model,
                 latent_spaces: List[LatentSpace],
                 reconstruction_loss,
                 activation="relu",
                 stop_gradient=False,
                 **kwargs):

        super(TransformVAE, self).__init__(**kwargs)


        assert len(encoders) == len(latent_spaces), "list of encoders and latent_spaces must have equal length"
        self._encoders = encoders
        self._decoder = decoder
        self.latent_spaces = latent_spaces
        self.reconstruction_loss = reconstruction_loss
        self.stop_gradient = stop_gradient

        self.x_in_shape = K.int_shape(encoders[0].input)[1:]
        for i, encoder in enumerate(encoders):
            assert K.int_shape(encoder.input)[1:] == self.x_in_shape, f"encoder {i} has different input shape"
            assert len(encoder.output.shape[1:]) == 1, "encoder {} output must be flattened," \
                                                       "i.e. have shape (batch_size, dim)"


        # Loss trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.kl_weight_tracker = tf.keras.metrics.Mean(name="kl_weight")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def set_lst_parameter_encoders(self) -> tf.keras.Model:
        lst_encoder_loc = []
        lst_encoder_scale = []
        # Define encoder backbone
        input_encoder = tf.keras.layers.Input(self.x_in_shape)

        for encoder_backbone, latent_space in zip(self._encoders, self.latent_spaces):
            h_enc = encoder_backbone(input_encoder)
            lst_encoder_loc.append(tf.keras.Model(input_encoder, latent_space.loc_param_layer(h_enc)))
            lst_encoder_scale.append(tf.keras.Model(input_encoder, latent_space.scale_param_layer(h_enc)))

        return lst_encoder_loc, lst_encoder_scale

    def set_decoder(self):
        h_dec = self._decoder.input
        assert len(h_dec.shape[1:]) == 1, "decoder input must be flattened, i.e. have shape (batch_size, dim)"
        units = int(h_dec.shape[1])
        dec_in_layer = Dense(units, activation=self.activation, name="dec_in")  # layer must be used in all models

        # SETUP DECODER
        # make lists of inputs, needed to define decoder model
        latent_dim = sum([latent_space.latent_dim for latent_space in self.latent_spaces])
        input_decoder = Input((latent_dim,))

        decoder = tf.keras.Model(input_decoder, self._decoder(dec_in_layer(input_decoder)))

        # Pass multiple codes to decoder
        mult_input_layer = tf.keras.layers.Input((self.n_transforms, latent_dim))
        x = tf.keras.layers.TimeDistributed(decoder)(mult_input_layer)
        decoder = tf.keras.Model(mult_input_layer, x)
        return decoder

    def set_encoder_transformed(self, n_transforms):
        # Define the input
        mult_input_layer = tf.keras.layers.Input((n_transforms, *self.x_in_shape))

        # Pass each image through the encoder output (num_transformers, list_encoder_outputs)
        lst_sample = [] # List of samples per latent space
        lst_loc = [] # List of loc parameter per latent space
        lst_scale = [] # List of scale parameter per latent space
        lst_sample_avg = [] # List of averaged latent after inverse transform
        for num_latent_space, latent_space in enumerate(self.latent_spaces):
            # Estimate parameter tensors with shape (batch_size, n_transforms, param_shape)
            loc_param_estimate = tf.keras.layers.TimeDistributed(self.lst_encoder_loc[num_latent_space])(
                mult_input_layer)
            scale_param_estimate = tf.keras.layers.TimeDistributed(self.lst_encoder_scale[num_latent_space])(
                mult_input_layer)

            lst_loc.append(loc_param_estimate)
            lst_scale.append(scale_param_estimate)
            lst_sample.append(latent_space.sampling([lst_loc[-1], lst_scale[-1]]))


            # Transformations
            if latent_space.transformation_shape is not None:
                transformations = Input(shape=(n_transforms,) + latent_space.transformation_shape)


            # Apply to sample inverse transform
            z_sample_anchored = latent_space.inverse_transform_layer([lst_sample[-1], transformations])
            z_sample_anchored_avg = latent_space.avg_layer(z_sample_anchored)
            z_sample_avg = latent_space.transform_layer([z_sample_anchored_avg, transformations])
            if self.stop_gradient:
                z_sample_avg = Lambda(lambda z: K.stop_gradient(z))(z_sample_avg)
            lst_sample_avg.append(z_sample_avg)



        # Create encoder from z and y
        encoder = tf.keras.models.Model(mult_input_layer, [lst_loc, lst_scale, lst_sample])
        return encoder

    def setup_model(self, n_transformed_datapoints, model_name=""):
        # if n_transformed_datapoints==1, acts like a normal VAE (but for data shape (batch_size, 1, *dims))
        # model_name is appended to the metric names

        # INPUT LAYER
        x_in = Input(shape=(n_transformed_datapoints,) + self.x_in_shape)

        # SETUP ENCODER FOR EACH LATENT SPACE
        transformations_list = []
        # h_enc_list = []
        z_params_list = []
        z_sample_list = []
        z_sample_avg_list = []
        for encoder, ls in zip(self._encoders, self.latent_spaces):
            if ls.transformation_shape is not None:
                transformations = Input(shape=(n_transformed_datapoints,) + ls.transformation_shape)
                transformations_list.append(transformations)
            # Pass the encoder in each of the n_transformed_datapoints datapoints for a batch
            h_enc = TimeDistributed(encoder, name=f"h_enc_{n_transformed_datapoints}way_{ls.name}")(x_in)
            # h_enc_list.append(h_enc)
            z_params = ls.get_params(h_enc)
            z_params_list.append(z_params)
            z_sample = ls.sample_layer(z_params)
            z_sample_list.append(z_sample)

            #   x_in has shape (batch_size, n_transformed_datapoints, *x_in.shape)
            #   z_sample has shape (batch_size, n_transformed_datapoints, ls.latent_dim)
            #   transformations has shape (batch_size, n_transformed_datapoints, transformation_shape)
            #   The assumption is that [batch_id, :, ...] (so all n_transformed_datapoints data points for a particular
            #   batch ID) are obtained from the same "anchor" x_anchor, by transforming it with the transformation given
            #   by the input layer "transformations". So we want to enforce the same to hold for z_sample. So for each
            #   latent variable we can compute z_anchor by applying the inversed transformation (in latent space). We
            #   then ideally want each z_anchor to be the same, and correspond to the encoding of x_anchor.
            #   Typically, the first observation for each batch ID (so [batch_id, 0, ...]) is the anchor, so
            #   transformations[batch_id, 0, ...] should represent the identity transformation.
            if n_transformed_datapoints == 1:
                z_sample_avg_list.append(z_sample)
            else:
                z_sample_anchored = ls.inverse_transform_layer([z_sample, transformations])
                z_sample_anchored_avg = ls.avg_layer(z_sample_anchored)
                z_sample_avg = ls.transform_layer([z_sample_anchored_avg, transformations])
                if self.stop_gradient:
                    z_sample_avg = Lambda(lambda z: K.stop_gradient(z))(z_sample_avg)
                z_sample_avg_list.append(z_sample_avg)

        # encoder model for all parameters of all latent spaces (as a flat list, since Keras requires this)
        all_parameters_flat = [param for z_params in z_params_list for param in z_params]
        encoder_params = Model(x_in, all_parameters_flat)

        # encoder model that outputs a single sample from each latent (posterior) distribution
        encoder_samples = Model(x_in, z_sample_list)

        # SETUP DECODER
        # make lists of inputs, needed to define decoder model
        samples_in = [Input(batch_shape=K.int_shape(sample)) for sample in z_sample_list]

        # concatenate all flat samples into one tensor
        if len(samples_in) == 1:
            samples_concat = samples_in[0]
        else:
            samples_concat = Concatenate(name=f"samples_concat_full_{n_transformed_datapoints}way")(samples_in)
        dec_in = self.dec_in_layer(samples_concat)
        dec_out = TimeDistributed(self._decoder, name=f"dec_out_{n_transformed_datapoints}way")(dec_in)

        # create full decoder model, takes list of samples as input
        decoder = Model(samples_in, dec_out)

        # setup full model & losses
        x_out = decoder(z_sample_avg_list)
        if n_transformed_datapoints == 1:
            full_model = Model(x_in, x_out)
        else:
            full_model = Model([x_in, *transformations_list], x_out)

        kl_loss_list = []
        weighted_kl_loss_total = 0
        dist_to_avg_loss_list = []
        weighted_dist_to_avg_loss_total = 0
        for ls, z_params, z_sample, z_sample_avg in zip(self.latent_spaces, z_params_list,
                                                        z_sample_list, z_sample_avg_list):
            kl_loss = K.sum(ls.kl_loss(z_params), axis=1)
            kl_loss_list.append(kl_loss)
            weighted_kl_loss_total += ls.kl_weight * kl_loss
            if n_transformed_datapoints > 1:
                # dist_to_avg_loss = K.sum(ls.distance(z_sample, z_sample_avg), axis=1)
                # Changed the distance to the appropriate sum
                dist_to_avg_loss = tf.reduce_sum(ls.distance(z_sample, z_sample_avg), axis=1)
                dist_to_avg_loss_list.append(dist_to_avg_loss)
                weighted_dist_to_avg_loss_total += ls.dist_weight * dist_to_avg_loss

        def loss_function(fake_true, fake_pred):
            return self.reconstruction_loss(x_in, x_out) \
                   + weighted_kl_loss_total \
                   + weighted_dist_to_avg_loss_total
        loss_function.__name__ = f"loss_{model_name}"

        # metrics, to keep track of individual loss components during training
        metrics = [loss_function]

        def metric_reconstr(fake_true, fake_pred):
            return self.reconstruction_loss(x_in, x_out)
        metric_reconstr.__name__ = f"metric_reconstr_{model_name}"
        metrics.append(metric_reconstr)

        def make_metric_kl(_kl_loss, num):
            def metric_kl(fake_true, fake_pred):
                return _kl_loss
            metric_kl.__name__ = f"metric_kl_{model_name}_{num}"
            return metric_kl
        for i, kl_loss in enumerate(kl_loss_list):
            metrics.append(make_metric_kl(kl_loss, i))

        def metric_weighted_kl_total(fake_true, fake_pred):
            return weighted_kl_loss_total
        metric_weighted_kl_total.__name__ = f"metric_weighted_kl_{model_name}"
        metrics.append(metric_weighted_kl_total)

        if n_transformed_datapoints > 1:
            def make_metric_dist_to_avg(_dist_to_avg_loss, num):
                def metric_dist_to_avg(fake_true, fake_pred):
                    return _dist_to_avg_loss
                metric_dist_to_avg.__name__ = f"metric_dist_to_avg_{model_name}_{num}"
                return metric_dist_to_avg
            for i, dist_to_avg_loss in enumerate(dist_to_avg_loss_list):
                metrics.append(make_metric_dist_to_avg(dist_to_avg_loss, i))

            def metric_weighted_dist_to_avg_total(fake_true, fake_pred):
                return weighted_dist_to_avg_loss_total
            metric_weighted_dist_to_avg_total.__name__ = f"metric_weighted_dist_to_avg_total_{model_name}"
            metrics.append(metric_weighted_dist_to_avg_total)

        full_model.compile(loss=loss_function, optimizer="adam", metrics=metrics, experimental_run_tf_function=False)
        full_model.summary()

        model = {
            "full_model": full_model,
            "encoder_params": encoder_params,
            "encoder_samples": encoder_samples,
            "decoder": decoder,
        }
        return model

    def setup_semi_supervised_models(self, n_transformed_datapoints):
        # model for batches labelled with transformations:
        model_l = self.setup_model(n_transformed_datapoints, model_name="l")
        # model for unlabelled batches:
        model_u = self.setup_model(1, model_name="u")
        return model_l, model_u

    def train_semi_supervised(self, model_l, model_u, x_l, x_l_transformations, x_u, epochs, batch_size,
                              callback_list=None):
        # assume x_l has shape (n_batches, n_transformed_datapoints, *data_shape):
        n_transformed_datapoints = x_l.shape[1]
        batch_size_l = batch_size // n_transformed_datapoints  # // to ensure integer batch size

        total_epochs = 0  # Total number of epochs = labeled epochs + unlabeled epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            if len(x_l) > 0:  # don't train if there is no labelled data
                print("Labelled training")
                model_l["full_model"].fit([x_l, *x_l_transformations], x_l,
                                          batch_size=batch_size_l, epochs=total_epochs + 1,
                                          callbacks=callback_list,
                                          initial_epoch=total_epochs)
                total_epochs += 1
            if len(x_u) > 0:  # don't train if there is no unlabelled data
                print("Unlabelled training")
                model_u["full_model"].fit(x_u, x_u,
                                          batch_size=batch_size, epochs=total_epochs + 1,
                                          callbacks=callback_list,
                                          initial_epoch=total_epochs)
                total_epochs += 1

    def compute_metrics(self, data_class, representations_list):
        transformations_array = data_class.flat_factor_mesh_as_angles  # shape (n_data_points, n_factors)
        # move factor axis forward so for-loop can loop through each factor
        transformations_array = np.moveaxis(transformations_array, 1, 0)  # shape (n_factors, n_data_points)
        transformations_array = np.reshape(transformations_array,
                                           (data_class.n_factors, data_class.n_data_points, 1, 1))
        # cast to float32 to avoid tf error in ls.transform
        transformations_array = transformations_array.astype(np.float32)

        distances = []
        angular_distances = []
        for representations, ls, transformations in zip(representations_list, self.latent_spaces,
                                                        transformations_array):
            # transform representations back to anchor (first datapoint)
            repr_anchored = ls.transform((representations, transformations),
                                         inverse=True)  # (n_data_points, 1, latent_dim)
            # compute average/centroid for anchored representations
            repr_anchored = np.moveaxis(repr_anchored, 1, 0)  # shape (1, n_data_points, latent_dim)
            repr_avg = ls.average(
                repr_anchored)  # shape (1, n_data_points, latent_dim), averaged and repeated over axis=-2
            # compute average distance between centroid and each anchored representation
            distance = ls.distance(repr_anchored, repr_avg)  # shape (1, n_data_points)
            distance = np.mean(distance, axis=1)
            distances.append(distance)
            # NOTE: works only with HypersphericalLatentSpace
            angular_distance = ls.distance_angular(repr_anchored, repr_avg)  # shape (1, n_data_points)
            angular_distance = np.mean(angular_distance, axis=1)
            angular_distances.append(angular_distance)
        # combine distances for all latent spaces
        mean_distance = np.mean(np.concatenate(distances, axis=0))
        mean_angular_distance = np.mean(np.concatenate(angular_distances, axis=0))

        return mean_distance, mean_angular_distance

if __name__ == "__main__":
    import numpy as np
    from modules.latent_space2 import GaussianLatentSpace, HyperSphericalLatentSpace

    latent_spaces = [GaussianLatentSpace(dim=10), HyperSphericalLatentSpace(dim=1)]
    input_shape = [64, 64, 3]
    images = np.zeros([10, 5] + input_shape)
    encoder, decoder = encoder_decoder_dense(input_shape=input_shape)
    init_params = {"latent_spaces": latent_spaces,
                   "input_shape": input_shape,
                   "encoder_backbones": [encoder],
                   "decoder_backbone": decoder,
                   "n_transforms": 5,
                   "average_mask": [True, False], }
    ugvae = GVAE(**init_params)