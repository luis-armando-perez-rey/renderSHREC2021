import sys
import os
import time
import argparse
import numpy as np
import json
import tensorflow as tf
import tensorflow_addons as tfa
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# PROJECT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
print("Project path", PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from data.data_loader import load_factor_data
from modules.vae.architectures import get_encoder_decoder
from modules.vae.autoencoder_models import AEClass, AETL, AE
from modules.vae.vae_models import VAE, VAEClass, VAETL
from modules.vae.triplet_transformvae import TripletTransformVAE
from modules.vae.transformvae_model import TransformVAE
from modules.vae.reconstruction_losses import bernoulli_loss
from modules.latent_space.latentspace2 import GaussianLatentSpace, HyperSphericalLatentSpace
from modules.utils import shrec_utils, shrec_evaluation
from tensorflow.python.framework.ops import disable_eager_execution


AVAILABLE_MODELS = ["AE", "AEclass", "AETL", "VAE", "VAEclass", "VAETL", "LSBDVAETL", "TLtd", "GVAE", "GVAETL", "GAE",
                    "GAETL", "LSBDVAE", "GVAEClass"]
AVAILABLE_ARCHITECTURES = ["dense", "vgg", "resnet50v2_dense", "resnet50v2_vgg"]

# ---------------------
# PARSE ARGUMENTS
# ---------------------
parser = argparse.ArgumentParser(description='Run train_model-like experiments')

parser.add_argument('--gpu', nargs="?", dest='gpu_num', type=str, default="0", help="gpu to be used")
# Data parameters
parser.add_argument('--collection_list', nargs="?", dest='collection_list', type=str,
                    help='shrec collection_list to pull data from')
parser.add_argument('--architecture', nargs="?", dest='architecture', type=str, default="resnet50v2_dense",
                    help='encoder decoder architecture to be used')
parser.add_argument('--dataset', nargs="?", dest='dataset', type=str, default="shrec2021", help='dataset used')
parser.add_argument('--challenge', nargs="?", dest='challenge', type=str, default="Shape", help='challenge dataset')
parser.add_argument('--testsplit', nargs="?", dest='test_split', type=float, default=0.3,
                    help="percentage of data that goes to test")
parser.add_argument('--selectedviews', nargs="?", dest='selected_views', type=int, default=12,
                    help="number of views")

# Model parameters
parser.add_argument('--modeltype', nargs="?", dest='model_type', type=str, default="AE",
                    help=f"model to be trained available options are {AVAILABLE_MODELS}")
parser.add_argument('--latentdim', nargs="?", dest='latent_dim', type=int, default=100,
                    help="latent dimension of Euclidean space")
parser.add_argument('--klweight', nargs="?", dest='kl_weight', type=float, default=1.0, help="kl weight of VAE models")
parser.add_argument('--normalize', nargs="?", dest='normalize', type=bool, default=False, help="normalize embeddings")
parser.add_argument('--class_alpha', nargs="?", dest='class_alpha', type=float, default=10000,
                    help="weighting parameter for auxiliary loss term (classification loss or triplet loss)")

# Training parameters
parser.add_argument('--classweights', nargs="?", dest='cw_bool', type=bool, default=False,
                    help="whether to add class weights to training")
parser.add_argument('--epochs', nargs="?", dest='epochs', type=int, default=100, help="training epochs")
parser.add_argument('--batchsize', nargs="?", dest='batch_size', type=int, default=50, help="batch size")

# Make submission
parser.add_argument('--submission', nargs="?", dest="submission", type=int, default=0)
parser.add_argument('--submissiontag', nargs="?", dest="submission_tag", type=str, default="")

# ---------------------
# READ PARAMETERS
# ---------------------
args = parser.parse_args()
print("Input arguments", args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
# Data parameters
selected_views = args.selected_views
challenge = args.challenge
dataset = args.dataset
test_split = args.test_split

# Model parameters
architecture = args.architecture
latent_dim = args.latent_dim
l_dim = 2
model_type = args.model_type
if model_type == "LSBDVAE":
    disable_eager_execution()
normalize = args.normalize

kl_weight = args.kl_weight
class_alpha = args.class_alpha
print(repr(architecture))
assert architecture in AVAILABLE_ARCHITECTURES, f"Architecture {architecture} not available. Available options are {AVAILABLE_ARCHITECTURES}"
assert model_type in AVAILABLE_MODELS, f"Model type {model_type} not available. Available options are {AVAILABLE_MODELS}"
print(f"Running {model_type} model.")

# Training parameters
epochs = args.epochs
batch_size = args.batch_size
cw_bool = args.cw_bool

# Submission parameters
submission = bool(args.submission)
submission_tag = args.submission_tag
if submission:
    test_split = 0.0

parameters = {}

# --------------------
# Data loading
# --------------------
if dataset == "shrec2021":
    data_parameters = {"data": "shrec2021",
                       "root_path": os.path.join(PROJECT_PATH, "data"),
                       "collection_list": [challenge],
                       "dataset_directory":"shrec2021",
                       "data_type": "train"}
    parameters.update(data_parameters)
    dataset_class = load_factor_data(**data_parameters)
    dataset_class.images = dataset_class.images[:, :selected_views, ...]
    dataset_class.labels = dataset_class.labels[:, :selected_views, ...]
    parameters.update({"num_views": selected_views})
    print("Shape of dataset images ", dataset_class.images.shape)
    n_views = dataset_class.images.shape[1]

    # Split the data for training, test and validation
    splitting_parameters = {"test_split": test_split,
                            "val_split": 0.0,
                            "random_state": 0,
                            "stratify": True}

    # Split data for training per object
    data_dictionary_list = shrec_utils.train_test_val_split([dataset_class.images, dataset_class.labels],
                                                            **splitting_parameters)


if submission:
    print("Submission is selected")
    data_parameters = {"data": "shrec2021",
                       "root_path": os.path.join(PROJECT_PATH, "data"),
                       "collection_list": [challenge],
                       "dataset_directory": "shrec2021",
                       "data_type": "test"}
    parameters.update(data_parameters)
    dataset_class = load_factor_data(**data_parameters)
    dataset_class.images = dataset_class.images[:, :selected_views, ...]
    data_dictionary_list[0]["test"] = dataset_class.images

# Get the image shape
image_shape = dataset_class.image_shape
# Remove dataset_class to avoid having too much memory used
del dataset_class

# Dictionary of number of objects per dataset
object_dictionary = shrec_utils.get_dictionary_object_quantities(data_dictionary_list[0])

# Flatten data for training per image
flat_dictionary_list = [shrec_utils.flatten_data_dictionary(data_dictionary) for data_dictionary in
                        data_dictionary_list]

# Transform class_labels to onehot
onehot_dictionary = shrec_utils.make_label_dictionary_onehot(flat_dictionary_list[1])

labels_dictionary_object = shrec_utils.get_labels_per_object_dictionary(flat_dictionary_list[1], object_dictionary,
                                                                        n_views)
# Create transformations for LSBDVAE
num_images = len(data_dictionary_list[0]["train"])
views = 2 * np.pi * np.linspace(0, 1, n_views, endpoint=False)
transformations_circular = np.expand_dims(np.array([views] * num_images), -1)
transformations_euclidean = np.ones(transformations_circular.shape)
transformations = [transformations_circular, transformations_euclidean]

# --------------------
# Model Architecture
# --------------------
# Architecture
parameters.update({"architecture": architecture})
tf.keras.backend.clear_session()
encoder_backbone, decoder_backbone = get_encoder_decoder(architecture, image_shape)

parameters.update({"model_type": model_type})
if model_type == "AE":
    model_parameters = {"dim": latent_dim,
                        "input_shape": image_shape,
                        "normalize": normalize}
    model_class = AE(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                     **model_parameters)
elif model_type == "AEclass":
    model_parameters = {"dim": latent_dim,
                        "input_shape": image_shape,
                        "normalize": normalize,
                        "num_labels": len(np.unique(data_dictionary_list[1]["train"]))}
    model_class = AEClass(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                                            **model_parameters)


elif model_type == "AETL":
    model_parameters = {"dim": latent_dim,
                        "input_shape": image_shape,
                        "normalize": normalize,
                        "class_alpha": class_alpha}
    model_class = AETL(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                                         **model_parameters)
elif model_type == "VAE":
    model_parameters = {"dim": latent_dim,
                        "input_shape": image_shape,
                        "normalize": normalize,
                        "kl_weight": kl_weight}

    model_class = VAE(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone, **model_parameters)
    model_class.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
elif model_type == "VAEclass":
    model_parameters = {"dim": latent_dim,
                        "input_shape": image_shape,
                        "kl_weight": kl_weight,
                        "normalize": normalize,
                        "num_labels": len(np.unique(data_dictionary_list[1]["train"])),
                        "class_alpha": class_alpha}

    model_class = VAEClass(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone, **model_parameters)
elif model_type == "VAETL":
    model_parameters = {"dim": latent_dim,
                        "input_shape": image_shape,
                        "kl_weight": kl_weight,
                        "normalize": normalize,
                        "class_alpha": class_alpha}

    model_class = VAETL(encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                                 **model_parameters)
    model_class.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))

elif model_type == "LSBDVAETL":
    latent_space_parameters = {"dim": latent_dim,
                               "kl_weight": kl_weight}
    parameters.update(latent_space_parameters)
    circular_latent_parameters = {"dim": 1, "steps": 10,
                                  "log_t_limit": (-10, -8),
                                  "kl_weight": kl_weight,
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
    model_dictionary = model_class.setup_model(n_transformed_datapoints=n_views, model_name="LSBDVAETL")

elif model_type == "LSBDVAE":
    latent_space_parameters = {"dim": latent_dim,
                               "kl_weight": kl_weight}
    parameters.update(latent_space_parameters)
    circular_latent_parameters = {"dim": 1, "steps": 10,
                                  "log_t_limit": (-10, -8),
                                  "kl_weight": kl_weight,
                                  "dist_weight": 1,
                                  "name": "circular"
                                  }
    circular_latent = HyperSphericalLatentSpace(**circular_latent_parameters)
    latent_space = GaussianLatentSpace(**latent_space_parameters)
    reconstruction_loss = bernoulli_loss()
    # reconstruction_loss = tf.keras.losses.BinaryCrossentropy
    model_parameters = {}
    model_class = TransformVAE(encoders=[encoder_backbone, encoder_backbone],
                               decoder=decoder_backbone,
                               latent_spaces=[circular_latent, latent_space],
                               reconstruction_loss=reconstruction_loss,
                               )
    model_dictionary = model_class.setup_model(n_transformed_datapoints=n_views, model_name="LSBDVAE")

elif model_type == "TLtd":
    model_parameters = {"dim": latent_dim}
    input_layer = tf.keras.layers.Input((n_views, *image_shape))
    h_out = tf.keras.layers.TimeDistributed(encoder_backbone)(input_layer)
    h_out = tf.keras.layers.Lambda(lambda y: tf.math.reduce_mean(y, axis=1))(h_out)
    h_out = tf.keras.layers.Dense(model_parameters["dim"])(h_out)
    encoder = tf.keras.models.Model(input_layer, h_out)
    encoder.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                    loss=tfa.losses.TripletSemiHardLoss())
else:
    model_class = None
    model_parameters = {}
    print(model_type)
    print(f"Model type {model_type} not available. Available models are {AVAILABLE_MODELS}")

parameters.update(model_parameters)

if model_type == "LSBDVAETL" or model_type == "LSBDVAE":
    encoder = model_dictionary["encoder_params"]
    decoder = model_dictionary["decoder"]
    train_model = model_dictionary["full_model"]
elif model_type == "TLtd":
    train_model = encoder
elif model_type == "VAETL" or model_type == "VAE" or model_type == "GVAE" or model_type == "GVAETL" or model_type == "GAE" or model_type == "GAETL" or model_type == "GVAEClass":
    encoder = model_class.encoder
    decoder = model_class.decoder
    train_model = model_class
else:
    encoder = model_class.encoder
    decoder = model_class.decoder
    train_model = model_class.set_autoencoder()
encoder_backbone.summary()
decoder_backbone.summary()

save_path = os.path.join(PROJECT_PATH, "results", "shrec2021", model_type)
os.makedirs(save_path, exist_ok=True)


model_name = model_type + "_" + architecture + "_" + challenge

training_parameters = {"batch_size": batch_size,
                       "epochs": epochs,
                       "verbose": 2}
parameters.update(training_parameters)



# Class weights
if cw_bool:
    if model_type == "TLtd" or model_type == "LSBDVAETL":
        class_weights = shrec_utils.get_dictionary_class_weights(labels_dictionary_object)["train"]

    else:
        class_weights = shrec_utils.get_dictionary_class_weights(flat_dictionary_list[1])["train"]
else:
    class_weights = None
parameters.update({"class_weights": cw_bool})


start_time = time.time()
if model_type == "AEclass" or model_type == "VAEclass" :
    x = flat_dictionary_list[0]["train"]
    y = [flat_dictionary_list[0]["train"], onehot_dictionary["train"]]

elif model_type == "AETL" or model_type == "VAETL":
    x = flat_dictionary_list[0]["train"]
    y = [flat_dictionary_list[0]["train"], flat_dictionary_list[1]["train"]]


elif model_type == "LSBDVAETL":
    x = [data_dictionary_list[0]["train"], *transformations]
    y = [data_dictionary_list[0]["train"], labels_dictionary_object["train"]]

elif model_type == "LSBDVAE":
    x = [data_dictionary_list[0]["train"], *transformations]
    y = data_dictionary_list[0]["train"]

elif model_type == "TLtd":
    x = data_dictionary_list[0]["train"]
    y = labels_dictionary_object["train"]
else:
    x = flat_dictionary_list[0]["train"]
    y = None

train_model.fit(x=x,
                y=y,
                callbacks=[],
                sample_weight=class_weights,
                **training_parameters)
training_time = time.time() - start_time
print("Training time --- %s seconds ---" % training_time)

# Saving models
if model_type not in ["LSBDVAETL", "LSBDVAE"]:
    encoder.save(os.path.join(save_path, model_name + "_encoder"))
    if model_type != "TLtd":
        decoder.save(os.path.join(save_path, model_name + "_decoder"))

# Encode data
start_time = time.time()
if model_type == "LSBDVAETL" or model_type == "LSBDVAE":
    embeddings_dictionary = shrec_utils.predict_parameters_lsbd_dictionary(encoder, data_dictionary_list[0])
    embeddings_dictionary_object = shrec_utils.embeddings_object_lsbd_dictionary(embeddings_dictionary)
    # Reshape embeddings dictionary for LSBDVAETL to only use Euclidean embeddings for plots and object identification
    for key in embeddings_dictionary.keys():
        if len(embeddings_dictionary[key]) != 0:
            embeddings_dictionary.update({key: embeddings_dictionary[key].reshape(
                (len(embeddings_dictionary[key]) * n_views, *embeddings_dictionary[key].shape[2:]))})
elif model_type == "TLtd":
    embeddings_dictionary = {"train": [], "test": [], "val": []}
    embeddings_dictionary_object = shrec_utils.make_prediction_dictionary(encoder, data_dictionary_list[0])

else:
    embeddings_dictionary = shrec_utils.make_prediction_dictionary(encoder, flat_dictionary_list[0])
    print("Embeddings shape", embeddings_dictionary["train"])
    embeddings_dictionary_object = shrec_utils.get_embeddings_per_object_dictionary(embeddings_dictionary,
                                                                                    object_dictionary, n_views)
encoding_time = time.time() - start_time
print("Encoding time --- %s seconds ---" % encoding_time)

labels_dictionary_object = shrec_utils.get_labels_per_object_dictionary(flat_dictionary_list[1],
                                                                        object_dictionary, n_views)

# Reconstruct data
if model_type != "TLtd":
    if model_type == "AEclass" or model_type == "AETL":
        reconstructions = train_model.predict(flat_dictionary_list[0]["train"])[0]
        reconstructions = reconstructions.reshape(
            (len(reconstructions) // n_views, n_views, *reconstructions.shape[1:]))
    elif model_type == "LSBDVAETL":
        reconstructions = train_model.predict([data_dictionary_list[0]["train"], *transformations])[0]
    elif model_type == "LSBDVAE":
        reconstructions = train_model.predict([data_dictionary_list[0]["train"], *transformations])
    else:
        reconstructions = train_model.predict(flat_dictionary_list[0]["train"])
        reconstructions = reconstructions.reshape(
            (len(reconstructions) // n_views, n_views, *reconstructions.shape[1:]))
    print(data_dictionary_list[0]["train"].shape, reconstructions.shape)
    fig, axes = shrec_utils.plot_reconstructions(data_dictionary_list[0]["train"], reconstructions, 0, n_views)


# Log PCA embeddings plots
if model_type not in ["TLtd"]:
    fig_dictionary = shrec_utils.fig_pca_dictionary(embeddings_dictionary, flat_dictionary_list[1])
else:
    fig_dictionary = shrec_utils.fig_pca_dictionary(embeddings_dictionary_object, labels_dictionary_object)


# Evaluate object wise
start_time = time.time()
if submission:
    print("Creating submission with {} query and {} collection".format(len(embeddings_dictionary_object["test"]),
                                                                       len(embeddings_dictionary_object["train"])))
    shrec_evaluation_class = shrec_evaluation.ShrecEmbeddingDistance(
        embeddings_dictionary_object["test"],
        embeddings_dictionary_object["train"],
        shrec_evaluation.l2_distance,
    )
    submission_path = os.path.join(save_path, "submission_matrix")
    os.makedirs(submission_path, exist_ok=True)
    shrec_evaluation_class.save_submission(
        filepath=os.path.join(submission_path, model_type + "_" + challenge + "_" + submission_tag + ".txt"))
else:
    shrec_evaluation_class = shrec_evaluation.ShrecEvaluation(
        embeddings_dictionary_object["test"],
        embeddings_dictionary_object["train"],
        shrec_evaluation.l2_distance,
        labels_dictionary_object["test"],
        labels_dictionary_object["train"],
    )
evaluation_time = time.time() - start_time
print("Evaluation time --- %s seconds ---" % evaluation_time)

time_dictionary = {"training_time": training_time,
                   "encoding_time": encoding_time,
                   "evaluation_time": evaluation_time}
if submission:
    time_path_file = os.path.join(submission_path,
                                  model_type + "_" + challenge + "_" + submission_tag + "_time.txt")
    with open(time_path_file, 'w') as outfile:
        json.dump(time_dictionary, outfile)

if not submission:
    # Plot true positive and confusion matrix
    fig, _ = shrec_evaluation.plot_matrix(shrec_evaluation_class.true_positive_matrix, "True Positive Matrix")
    fig = shrec_utils.plot_confusion_matrix(labels_dictionary_object["test"], shrec_evaluation_class.prediction)
