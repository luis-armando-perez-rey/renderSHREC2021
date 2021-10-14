import sys
import os
import time
import argparse
import numpy as np
import tensorflow as tf
import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor

PROJECT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(PROJECT_PATH)

from modules.utils import shrec_utils, shrec_evaluation
from modules.vae.architectures import get_encoder_decoder
from data.grouped_dataset import GroupedDataset
from experiments.shrec2021.model_utils import load_model, get_reconstructions, get_embeddings_dictionary
from tensorflow.python.framework.ops import disable_eager_execution

from experiments import neptune_config

AVAILABLE_MODELS = ["AE", "AEClass", "AETL", "VAE", "VAEClass", "VAETL", "LSBDVAETL", "TLtd",
                    "GVAE", "GVAETL", "GVAEClass", "GVAEE", "GVAETLE", "GVAEClassE",
                    "GAE", "GAETL", "GAEClass", "GAEE", "GAETLE", "GAEClassE", "GAEProto", "GAEProtoE"
                                                                                           "LSBDVAE", ]
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
parser.add_argument('--latentdims', nargs="+", dest='latent_dims', type=int, default=[100, 0],
                    help="latent dimensions of latent spaces space")
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
parser.add_argument('--submission', nargs="?", dest="submission", type=bool, default=False)
parser.add_argument('--submissiontag', nargs="?", dest="submission_tag", type=str, default="")

# ---------------------
# READ PARAMETERS
# ---------------------
args = parser.parse_args()
print("Input arguments", args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

if args.model_type == "LSBDVAE":
    disable_eager_execution()

assert args.architecture in AVAILABLE_ARCHITECTURES, f"Architecture {args.architecture} not available. Available options are {AVAILABLE_ARCHITECTURES}"
assert args.model_type in AVAILABLE_MODELS, f"Model type {args.model_type} not available. Available options are {AVAILABLE_MODELS}"
print(f"Running {args.model_type} model.")

submission = args.submission
submission_tag = args.submission_tag

parameters = {}

# --------------------
# Data loading
# --------------------
dataset_parameters = {"dataset": args.dataset,
                      "selected_views": args.selected_views,
                      "test_split": args.test_split}
parameters.update(dataset_parameters)
dataset_class = GroupedDataset(**dataset_parameters)


# Transform class_labels to onehot

# Create transformations for LSBDVAE
num_images = len(dataset_class.data_dictionary_list[0]["train"])
views = 2 * np.pi * np.linspace(0, 1, args.selected_views, endpoint=False)
transformations_circular = np.expand_dims(np.array([views] * num_images), -1)
transformations_euclidean = np.ones(transformations_circular.shape)
transformations = [transformations_circular, transformations_euclidean]

# --------------------
# Model Architecture
# --------------------
# Architecture
parameters.update({"architecture": args.architecture})
tf.keras.backend.clear_session()
encoder_backbone, decoder_backbone = get_encoder_decoder(args.architecture, dataset_class.image_shape)

model_parameters = {"model_type": args.model_type,
                    "latent_dims": args.latent_dims,
                    "kl_weight": args.kl_weight,
                    "class_alpha": args.class_alpha,
                    "normalize": args.normalize}
model_class = load_model(data_class=dataset_class, encoder_backbone=encoder_backbone, decoder_backbone=decoder_backbone,
                         **model_parameters)
model_class.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
parameters.update(model_parameters)
encoder_backbone.summary()
decoder_backbone.summary()

save_path = os.path.join(PROJECT_PATH, "results", "shrec2021", args.model_type)
os.makedirs(save_path, exist_ok=True)

model_name = args.model_type + "_" + args.architecture + "_" + args.dataset

training_parameters = {"batch_size": args.batch_size,
                       "epochs": args.epochs,
                       "verbose": 2}
parameters.update(training_parameters)

# Neptune
group = "TUe"
api_token = neptune_config.API_KEY
neptune.init(project_qualified_name=group + "/shrec2021", api_token=api_token)

with neptune.create_experiment(model_name, params=parameters):
    start_time = time.time()
    # Get training data and class weights
    x, y = dataset_class.get_training_xy(model_class, batch_size=args.batch_size)
    if args.cw_bool:
        class_weights = dataset_class.get_class_weights_dictionary_model(args.model_type)["train"]
    else:
        class_weights = None
    parameters.update({"class_weights": args.cw_bool})

    if dataset_class == "cub_dataset_transforms":
        # Train model
        model_class.fit(x=x,
                        callbacks=[NeptuneMonitor(), tf.keras.callbacks.TerminateOnNaN()],
                        sample_weight=class_weights,
                        **training_parameters)
    else:
        # Train model
        model_class.fit(x=x,
                        y=y,
                        callbacks=[NeptuneMonitor(), tf.keras.callbacks.TerminateOnNaN()],
                        sample_weight=class_weights,
                        **training_parameters)
    training_time = time.time() - start_time
    print("Training time --- %s seconds ---" % training_time)

    # Save model
    model_class.save_weights(os.path.join(save_path, model_name + ".h5"))

    # Encode data
    start_time = time.time()
    embeddings_dictionary_image, embeddings_dictionary_object = get_embeddings_dictionary(args.model_type, model_class,
                                                                                          dataset_class)
    encoding_time = time.time() - start_time
    print("Encoding time --- %s seconds ---" % encoding_time)

    # Reconstruct data
    reconstructions = get_reconstructions(args.model_type, dataset_class, model_class)
    fig, axes = shrec_utils.plot_reconstructions(dataset_class.data_dictionary_list[0]["train"], reconstructions, 0,
                                                 dataset_class.selected_views)
    neptune.log_image('plots', fig, image_name='reconstructions')

    # Log PCA embeddings plots
    print("Labels shape ", dataset_class.flat_dictionary_list[1]["train"].shape)
    print("Embeddings shape ", embeddings_dictionary_image["train"].shape)
    # fig_dictionary = shrec_utils.fig_pca_dictionary(embeddings_dictionary_image, dataset_class.flat_dictionary_list[1])
    # shrec_utils.neptune_log_fig_dictionary(fig_dictionary, base_name="image_pca")

    # Plots at object level
    # fig_dictionary = shrec_utils.fig_pca_dictionary(embeddings_dictionary_object, dataset_class.data_dictionary_list[2])
    # shrec_utils.neptune_log_fig_dictionary(fig_dictionary, base_name="object_pca")

    # Evaluate object wise
    start_time = time.time()

    shrec_evaluation_class = shrec_evaluation.ShrecEvaluation(
        embeddings_dictionary_object["test"],
        embeddings_dictionary_object["train"],
        shrec_evaluation.l2_distance,
        dataset_class.data_dictionary_list[2]["test"],
        dataset_class.data_dictionary_list[2]["train"],
    )
    evaluation_time = time.time() - start_time
    print("Evaluation time --- %s seconds ---" % evaluation_time)

    time_dictionary = {"training_time": training_time,
                       "encoding_time": encoding_time,
                       "evaluation_time": evaluation_time}

    # Neptune log metrics
    shrec_utils.neptune_log_metric_dictionary(shrec_evaluation_class.metric_dictionary, "object")

    # Plot true positive and confusion matrix
    fig, _ = shrec_evaluation.plot_matrix(shrec_evaluation_class.true_positive_matrix, "True Positive Matrix")
    neptune.log_image('plots', fig, image_name='object true positive matrix')
    fig = shrec_utils.plot_confusion_matrix(dataset_class.data_dictionary_list[2]["test"],
                                            shrec_evaluation_class.prediction)
    neptune.log_image('plots', fig, image_name='object confusion matrix')
