import os
import sys
import numpy as np
import tensorflow as tf
import json
from typing import List, Dict, NoReturn

PROJECT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(PROJECT_PATH)
from modules.latent_space.latentspace2 import GaussianLatentSpace, HyperSphericalLatentSpace
from modules.vae.autoencoder_models import AE, AETL, AEClass
from modules.vae.gae_models import GAE, GAETL, GAEClass, GAEProto
from modules.vae.gvae_models import GVAE, GVAETL, GVAEClass, GVAEProto
from modules.vae.reconstruction_losses import bernoulli_loss
from modules.vae.transformvae import TransformVAE
from modules.vae.triplet_transformvae import TripletTransformVAE
from modules.vae.vae_models import VAE, VAEClass, VAETL
from modules.vae.tl_models import TL, TLo
from experiments.shrec2021.dataset_utils import GroupedDataset
from modules.utils import shrec_utils


def save_parameters(parameters: Dict,
                    save_path: str) -> NoReturn:
    with open(save_path, "w+") as f:
        json.dump(parameters, f)


def load_parameters(load_path: str) -> Dict:
    with open(load_path) as f:
        parameters = json.load(f)
    return parameters


def load_model(data_class: GroupedDataset,
               model_type: str,
               latent_dims: List,
               encoder_backbone: tf.keras.Model,
               decoder_backbone: tf.keras.Model,
               kl_weight: List[float] = [1.0],
               class_alpha: float = 1.0,
               normalize=False):
    total_latent_dim = int(np.sum(latent_dims))
    total_kl_weight = np.sum(kl_weight)
    if model_type == "AE":
        model_parameters = {"dim": total_latent_dim,
                            "input_shape": data_class.image_shape,
                            "normalize": normalize}
        model_class = AE(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                         **model_parameters)
    elif model_type == "AEClass":
        model_parameters = {"dim": total_latent_dim,
                            "input_shape": data_class.image_shape,
                            "normalize": normalize,
                            "class_alpha": class_alpha,
                            "num_labels": len(np.unique(data_class.data_dictionary_list[1]["train"]))}
        model_class = AEClass(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                              **model_parameters)
    elif model_type == "AETL":
        model_parameters = {"dim": total_latent_dim,
                            "input_shape": data_class.image_shape,
                            "normalize": normalize,
                            "class_alpha": class_alpha}
        model_class = AETL(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                           **model_parameters)

    elif model_type == "VAE":
        model_parameters = {"dim": total_latent_dim,
                            "input_shape": data_class.image_shape,
                            "normalize": normalize,
                            "kl_weight": total_kl_weight}

        model_class = VAE(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone, **model_parameters)
    elif model_type == "VAEClass":
        model_parameters = {"dim": total_latent_dim,
                            "input_shape": data_class.image_shape,
                            "kl_weight": total_kl_weight,
                            "normalize": normalize,
                            "num_labels": len(np.unique(data_class.data_dictionary_list[1]["train"])),
                            "class_alpha": class_alpha}

        model_class = VAEClass(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone, **model_parameters)
    elif model_type == "VAETL":
        model_parameters = {"dim": total_latent_dim,
                            "input_shape": data_class.image_shape,
                            "kl_weight": total_kl_weight,
                            "normalize": normalize,
                            "class_alpha": class_alpha}

        model_class = VAETL(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                            **model_parameters)
    elif model_type in ["GVAE", "GVAEE"]:
        circular_latent_parameters = {"dim": latent_dims[0], "steps": 10,
                                      "log_t_limit": (-10, -6),
                                      "kl_weight": kl_weight[0],
                                      "dist_weight": 1
                                      }
        if model_type == "GVAE":
            latent_spaces = [HyperSphericalLatentSpace(**circular_latent_parameters), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {"kl_weight": kl_weight[1],
                            "n_transforms": data_class.selected_views,
                            "input_shape": data_class.image_shape,
                            "average_mask": [False, True]}
        model_class = GVAE(latent_spaces=latent_spaces, encoder_backbones=[encoder_backbone],
                           decoder_backbone=decoder_backbone,
                           **model_parameters)

    elif model_type in ["GVAEClass", "GVAEClassE"]:
        if model_type == "GVAEClass":
            latent_spaces = [HyperSphericalLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=0), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {"kl_weight": 1.0,
                            "n_transforms": data_class.selected_views,
                            "input_shape": data_class.image_shape,
                            "average_mask": [False, True],
                            "class_alpha": class_alpha,
                            "num_classes": len(np.unique(data_class.data_dictionary_list[1]["train"]))}
        print("Model parameters", model_parameters)
        model_class = GVAEClass(latent_spaces=latent_spaces,
                                encoder_backbones=[encoder_backbone],
                                decoder_backbone=decoder_backbone,
                                **model_parameters)

    elif model_type in ["GVAETL", "GVAETLE"]:
        if model_type == "GVAETL":
            latent_spaces = [HyperSphericalLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {"kl_weight": 1.0,
                            "class_alpha": class_alpha,
                            "n_transforms": data_class.selected_views,
                            "input_shape": data_class.image_shape,
                            "average_mask": [False, True]}
        model_class = GVAETL(latent_spaces=latent_spaces, encoder_backbones=[encoder_backbone],
                             decoder_backbone=decoder_backbone,
                             **model_parameters)
    elif model_type in ["GAE", "GAEE"]:
        if model_type == "GAE":
            latent_spaces = [HyperSphericalLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {
            "n_transforms": data_class.selected_views,
            "input_shape": data_class.image_shape,
            "average_mask": [False, True]}
        model_class = GAE(latent_spaces=latent_spaces, encoder_backbones=[encoder_backbone],
                          decoder_backbone=decoder_backbone,
                          **model_parameters)

    elif model_type in ["GAEClass", "GAEClassE"]:
        if model_type == "GAEClass":
            latent_spaces = [HyperSphericalLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[1]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {
            "n_transforms": data_class.selected_views,
            "input_shape": data_class.image_shape,
            "average_mask": [False, True],
            "class_alpha": class_alpha,
            "num_classes": len(np.unique(data_class.data_dictionary_list[1]["train"]))}
        print("Model parameters", model_parameters)
        model_class = GAEClass(latent_spaces=latent_spaces,
                               encoder_backbones=[encoder_backbone],
                               decoder_backbone=decoder_backbone,
                               **model_parameters)
    elif model_type in ["GAEProto", "GAEProtoE"]:
        if model_type == "GAEProto":
            latent_spaces = [HyperSphericalLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {
            "n_transforms": data_class.selected_views,
            "input_shape": data_class.image_shape,
            "average_mask": [False, True],
            "class_alpha": class_alpha, }
        print("Model parameters", model_parameters)
        model_class = GAEProto(latent_spaces=latent_spaces,
                               encoder_backbones=[encoder_backbone],
                               decoder_backbone=decoder_backbone,
                               **model_parameters)
    elif model_type in ["GVAEProto", "GVAEProtoE"]:
        if model_type == "GVAEProto":
            latent_spaces = [HyperSphericalLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {
            "n_transforms": data_class.selected_views,
            "input_shape": data_class.image_shape,
            "average_mask": [False, True],
            "class_alpha": class_alpha, }
        print("Model parameters", model_parameters)
        model_class = GVAEProto(latent_spaces=latent_spaces,
                                encoder_backbones=[encoder_backbone],
                                decoder_backbone=decoder_backbone,
                                **model_parameters)


    elif model_type in ["GAETL", "GAETLE"]:
        if model_type == "GAETL":
            latent_spaces = [HyperSphericalLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        else:
            latent_spaces = [GaussianLatentSpace(dim=latent_dims[0], kl_weight=kl_weight[0]), GaussianLatentSpace(dim=latent_dims[1], kl_weight=kl_weight[1])]
        model_parameters = {
            "class_alpha": class_alpha,
            "n_transforms": data_class.selected_views,
            "input_shape": data_class.image_shape,
            "average_mask": [False, True]}
        model_class = GAETL(latent_spaces=latent_spaces, encoder_backbones=[encoder_backbone],
                            decoder_backbone=decoder_backbone,
                            **model_parameters)
    elif model_type == "LSBDVAETL":
        latent_space_parameters = {"dim": total_latent_dim,
                                   "kl_weight": kl_weight[1]}
        circular_latent_parameters = {"dim": 1, "steps": 10,
                                      "log_t_limit": (-10, -6),
                                      "kl_weight": kl_weight[0],
                                      "dist_weight": 1,
                                      "name": "circular"
                                      }
        circular_latent = HyperSphericalLatentSpace(**circular_latent_parameters)
        latent_space = GaussianLatentSpace(**latent_space_parameters)
        # reconstruction_loss = bernoulli_loss
        reconstruction_loss = tf.keras.losses.BinaryCrossentropy
        model_parameters = {"class_alpha": class_alpha}
        model_class = TripletTransformVAE(encoders=[encoder_backbone, encoder_backbone],
                                          decoder=decoder_backbone,
                                          latent_spaces=[circular_latent, latent_space],
                                          reconstruction_loss=reconstruction_loss(),
                                          **model_parameters)
    elif model_type == "LSBDVAE":
        latent_space_parameters = {"dim": total_latent_dim,
                                   "kl_weight": kl_weight[1]}
        circular_latent_parameters = {"dim": 1, "steps": 10,
                                      "log_t_limit": (-10, -6),
                                      "kl_weight": kl_weight[0],
                                      "dist_weight": 1,
                                      "name": "circular"
                                      }
        circular_latent = HyperSphericalLatentSpace(**circular_latent_parameters)
        latent_space = GaussianLatentSpace(**latent_space_parameters)
        reconstruction_loss = bernoulli_loss()

        model_class = TransformVAE(encoders=[encoder_backbone, encoder_backbone],
                                   decoder=decoder_backbone,
                                   latent_spaces=[circular_latent, latent_space],
                                   reconstruction_loss=reconstruction_loss,
                                   )
    elif model_type == "TLo":
        model_class = TLo(encoder_backbone=encoder_backbone,
                          latent_dim=total_latent_dim,
                          n_transforms=data_class.selected_views,
                          input_shape=data_class.image_shape
                          )
    elif model_type == "TL":
        model_class = TL(encoder_backbone=encoder_backbone,
                                            latent_dim=total_latent_dim,
                                            input_shape=data_class.image_shape
                                            )
    else:
        model_class = None
        print(model_type + " model doesn't exist")
    return model_class


def get_embeddings_dictionary(model_type, model, data_class: GroupedDataset):
    """
    Create embeddings
    Args:
        model_type:
        model:
        data_class:

    Returns:
        embeddings_dictionary_image: dictionary of embeddings with shape (num_image, latent_dim)
        embeddings_dictionary_object: dictionary of embeddings with shape (object_num, view_num, latent_dim)
    """
    print("Creating embeddings for", model_type)
    if model_type == "LSBDVAETL" or model_type == "LSBDVAE":
        embeddings_dictionary_image = shrec_utils.predict_parameters_lsbd_dictionary(model.encoder,
                                                                                     data_class.data_dictionary_list[0])
        embeddings_dictionary_object = shrec_utils.embeddings_object_lsbd_dictionary(embeddings_dictionary_image)
        # Reshape embeddings dictionary for LSBDVAETL to only use Euclidean embeddings for plots and object identification
        for key in embeddings_dictionary_image.keys():
            if len(embeddings_dictionary_image[key]) != 0:
                embeddings_dictionary_image.update({key: embeddings_dictionary_image[key].reshape(
                    (len(embeddings_dictionary_image[key]) * data_class.selected_views,
                     *embeddings_dictionary_image[key].shape[2:]))})

    elif model_type == "TLtd":
        embeddings_dictionary_image = {"train": [], "test": [], "val": []}
        embeddings_dictionary_object = shrec_utils.make_prediction_dictionary(model.encoder,
                                                                              data_class.data_dictionary_list[0])
    elif model_type in ["GVAE", "GVAETL", "GAE", "GAETL", "GVAEClass", "GAEProto", "GAEClass"]:

        # Make the embeddings for latent space Z
        prediction_function = model.make_embedding_function(-1)
        embeddings_dictionary_z = shrec_utils.apply_function_to_dictionary(prediction_function,
                                                                           data_class.data_dictionary_list[0])

        print("Size embeddings", embeddings_dictionary_z["train"].shape)
        if embeddings_dictionary_z["train"].shape[1] == 1:
            embeddings_dictionary_object = {key: np.squeeze(value) for key, value in
                                            embeddings_dictionary_z.items()}
        else:
            [print(value.shape) for key, value in embeddings_dictionary_z.items()]
            embeddings_dictionary_object = {key: np.mean(value, axis=1) for key, value in
                                            embeddings_dictionary_z.items()}
        embeddings_dictionary_image = shrec_utils.flatten_data_dictionary(embeddings_dictionary_z)
    else:
        embeddings_dictionary_image = shrec_utils.make_prediction_dictionary(model.encoder,
                                                                             data_class.flat_dictionary_list[0])
        print("Embeddings shape", embeddings_dictionary_image["train"])
        embeddings_dictionary_object = shrec_utils.get_embeddings_per_object_dictionary(embeddings_dictionary_image,
                                                                                        data_class.num_object_dictionary,
                                                                                        data_class.selected_views)

    return embeddings_dictionary_image, embeddings_dictionary_object


def get_reconstructions(model_type, data_class: GroupedDataset, train_model):
    print("Reconstructing data using {}".format(model_type))
    # Reconstruct data
    if model_type != "TL" and model_type != "TLo":
        if model_type == "AEClass" or model_type == "AETL":
            reconstructions = train_model.predict(data_class.flat_dictionary_list[0]["train"])[0]
            reconstructions = reconstructions.reshape(
                (len(reconstructions) // data_class.selected_views, data_class.selected_views,
                 *reconstructions.shape[1:]))
        elif model_type == "LSBDVAETL":
            reconstructions = \
                train_model.predict([data_class.data_dictionary_list[0]["train"], *data_class.transformations])[0]
        elif model_type == "LSBDVAE":
            reconstructions = train_model.predict(
                [data_class.data_dictionary_list[0]["train"], *data_class.transformations])
        elif model_type in ["GVAE", "GVAETL", "GAE", "GAETL", "GVAEClass", "GAEProto", "GVAEProto", "GAEClass"]:
            reconstructions = train_model.predict(data_class.data_dictionary_list[0]["train"])
            reconstructions = reconstructions.reshape(
                (len(reconstructions), data_class.selected_views, *reconstructions.shape[2:]))
        else:
            reconstructions = train_model.predict(data_class.flat_dictionary_list[0]["train"])
            reconstructions = reconstructions.reshape(
                (len(reconstructions) // data_class.selected_views, data_class.selected_views,
                 *reconstructions.shape[1:]))
    else:
        reconstructions = None
    return reconstructions
