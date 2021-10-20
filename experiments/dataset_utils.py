import sys
import os
import numpy as np
from typing import Dict
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(PROJECT_PATH)

from data_scripts.data_loader import load_factor_data
from modules.utils import shrec_utils


COMPLETE_COLLECTION = ["airplane",
                       "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
                       "car", "chair", "cone", "cup", "curtain",
                       "desk", "door", "dresser",
                       "flower_pot",
                       "glass_box",
                       "guitar",
                       "keyboard",
                       "lamp", "laptop",
                       "mantel", "monitor",
                       "night_stand",
                       "person", "piano", "plant",
                       "radio", "range_hood",
                       "sink", "sofa", "stairs", "stool",
                       "table", "tent", "toilet", "tv_stand",
                       "vase",
                       "wardrobe",
                       "xbox"]


def load_shrec2021_challenge(challenge, selected_views, test_split):
    data_parameters = {"data": "shrec2021",
                       "root_path": PROJECT_PATH,
                       "collection_list": [challenge],
                       "data_type": "train"}

    dataset_class = load_factor_data(**data_parameters)
    dataset_class.images = dataset_class.images[:, :selected_views, ...]
    dataset_class.labels = dataset_class.labels[:, :selected_views, ...]

    print("Shape of dataset images ", dataset_class.images.shape)

    # Split the data for training, test and validation
    splitting_parameters = {"test_split": test_split,
                            "val_split": 0.0,
                            "random_state": 0,
                            "stratify": True}

    # Split data for training per object
    data_dictionary_list = shrec_utils.train_test_val_split(
        [dataset_class.images, dataset_class.labels, dataset_class.labels[:, 0, ...]],
        **splitting_parameters)
    return data_dictionary_list


def select_mapping_function(selected_views, model_type=""):

    if model_type=="GAE"or model_type == "GVAE" or model_type == "GAEProto":
        def mapping_function(image, label):
            """
            This is the mapping that takes the tf.DataSet crops the images w.r.t bounding boxes and resizes them
            :param dictionary_:
            :return:
            """
            transformed_images = []
            for _ in range(selected_views-1):
                random_scale = tf.random.uniform(shape=(1,), minval=0.5, maxval=0.8, dtype=tf.dtypes.float32, )
                crop_resolution = tf.cast(random_scale * (64, 64), tf.int32)
                crop_size = (crop_resolution[0], crop_resolution[1], 3)
                transformed_images.append(
                    tf.expand_dims(tf.image.resize(tf.image.random_crop(image, size=crop_size), size=RESOLUTION), 0))
                # transformed_images.append(tf.expand_dims(image, axis=0))
            image = tf.concat([tf.expand_dims(image, axis=0)] + transformed_images, axis=0)
            return image
    elif model_type == "VAE" or model_type=="AE":
        def mapping_function(image, label):
            """
            This is the mapping that takes the tf.DataSet crops the images w.r.t bounding boxes and resizes them
            :param dictionary_:
            :return:
            """
            percentage_transform = 1/selected_views
            print("Percentage transform")
            bool_transform = tf.random.uniform(shape=(1,), minval = 0, maxval=1.0, dtype=tf.dtypes.float32,)


            # if bool_transform>=percentage_transform:
            random_scale = tf.random.uniform(shape=(1,), minval=0.5, maxval=0.8, dtype=tf.dtypes.float32, )
            crop_resolution = tf.cast(random_scale * (64, 64), tf.int32)
            crop_size = (crop_resolution[0], crop_resolution[1], 3)
            image = tf.image.resize(tf.image.random_crop(image, size=crop_size), size=RESOLUTION)


            def output_fn():
                return tf.image.resize(tf.image.random_crop(image, size=crop_size), size=RESOLUTION)
            def constant_fn():
                return image
            image = tf.cond(bool_transform >= percentage_transform,
                             output_fn,
                             constant_fn)
            return image
    else:
        def mapping_function(image, label):
            """
            This is the mapping that takes the tf.DataSet crops the images w.r.t bounding boxes and resizes them
            :param dictionary_:
            :return:
            """
            transformed_images = []
            for _ in range(selected_views-1):
                random_scale = tf.random.uniform(shape=(1,), minval=0.5, maxval=0.8, dtype=tf.dtypes.float32, )
                crop_resolution = tf.cast(random_scale * (64, 64), tf.int32)
                crop_size = (crop_resolution[0], crop_resolution[1], 3)
                # print(crop_resolution[0], crop_resolution[1])
                # transformed_images.append(tf.expand_dims(image, axis = 0))
                transformed_images.append(
                    tf.expand_dims(tf.image.resize(tf.image.random_crop(image, size=crop_size), size=RESOLUTION), 0))
            label = tf.concat([tf.expand_dims(label, axis = 0)]*selected_views, axis = 0)
            image = tf.concat([tf.expand_dims(image, axis=0)] + transformed_images, axis=0)
            return image, label
    return mapping_function


def segment_crop_norm_image(dictionary_):
    """
    This is the mapping that takes the tf.DataSet crops the images w.r.t bounding boxes and resizes them
    :param dictionary_:
    :return:
    """
    segmentation_value = 204
    resolution = [64, 64]
    image = dictionary_["image"]
    bbox = dictionary_["bbox"]
    mask = dictionary_["segmentation_mask"]
    label = dictionary_["label"]

    mask = tf.cast(mask >= segmentation_value, image.dtype)
    image = image * mask
    image = tf.image.crop_and_resize(tf.expand_dims(image, 0), [bbox], [0], resolution)[0]
    image = tf.cast(image, tf.float32) / 255.
    return image, label



def select_mapping_colored_backgrounds(selected_views):
    def segment_color_images(dictionary_):
        """
            This is the mapping that takes the tf.DataSet crops the images w.r.t bounding boxes and resizes them
            :param dictionary_:
            :return:
            """
        segmentation_value = 204
        resolution = [64, 64]
        image = dictionary_["image"]
        bbox = dictionary_["bbox"]
        mask = dictionary_["segmentation_mask"]
        label = dictionary_["label"]
        transformed_images = []
        for _ in range(selected_views):
            color = tf.random.uniform(minval=0, maxval=1, shape=(3,), dtype=tf.dtypes.float32) * 255
            # color = color.astype(int)
            segmentation_mask = tf.cast(mask >= segmentation_value, image.dtype)
            color_mask = tf.cast(mask < segmentation_value, image.dtype)
            color_mask = tf.concat([tf.cast(color[0] , image.dtype)* color_mask, tf.cast(color[1] , image.dtype) * color_mask, tf.cast(color[2] , image.dtype) * color_mask], axis=-1)
            transformed_image = image * segmentation_mask
            transformed_image = transformed_image + color_mask
            transformed_image = tf.image.crop_and_resize(tf.expand_dims(transformed_image, 0), [bbox], [0], resolution)[
                0]
            transformed_image = tf.expand_dims(tf.cast(transformed_image, tf.float32) / 255., axis=0)
            transformed_images.append(transformed_image)
        label = tf.concat([tf.expand_dims(label, axis=0)] * selected_views, axis=0)
        image = tf.concat(transformed_images, axis=0)
        return image, label
    return segment_color_images


def crop_norm_image(dictionary_):
    """
    This is the mapping that takes the tf.DataSet crops the images w.r.t bounding boxes and resizes them
    :param dictionary_:
    :return:
    """
    resolution = [64, 64]
    image = dictionary_["image"]
    bbox = dictionary_["bbox"]
    label = dictionary_["label"]

    image = tf.image.crop_and_resize(tf.expand_dims(image, 0), [bbox], [0], resolution)[0]
    image = tf.cast(image, tf.float32) / 255.
    return image, label




def load_tfds_data(ds):
    images = []
    labels = []
    for example in tfds.as_numpy(ds):
        images.append(example[0])
        labels.append(example[1])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def load_cub_dataset():
    (ds_train, ds_test), ds_info = tfds.load("caltech_birds2011", split=["train", "test"], with_info=True,
                                             data_dir="/data/downloads/manual")
    ds_train = ds_train.map(segment_crop_norm_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(segment_crop_norm_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    images_train, labels_train = load_tfds_data(ds_train)
    images_test, labels_test = load_tfds_data(ds_test)
    data_dictionary_list = [{"train":images_train, "test":images_test},
                            {"train":labels_train, "test":labels_test},
                            {"train":labels_train[:,0,...], "test":labels_test[:,0, ...]},]
    return data_dictionary_list


RESOLUTION = [64, 64]
segmentation_value = 204
max_delta = 0.5
num_batch_elements = 1


def load_tensorflow_dataset_transforms(name_dataset, model_type, batch_size, transforms):
    (ds_train, ds_test), ds_info = tfds.load(name_dataset , split=["train", "test"], with_info=True,
                                             data_dir="/data/downloads/manual")
    for transform in transforms:
        ds_train = ds_train.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # mapping_function = select_mapping_function(selected_views, ds_info, model_type)
    # ds_train = ds_train.map(segment_crop_norm_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_test = ds_test.map(segment_crop_norm_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.map(mapping_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_test = ds_test.map(mapping_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if model_type != "":
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test

def load_tf_dataset_dictionary_transforms(dataset_name, selected_views, transforms):
    ds_train, ds_test = load_tensorflow_dataset_transforms(dataset_name, model_type="", batch_size=1, transforms=transforms)
    images_train, labels_train = load_tfds_data(ds_train)
    images_test, labels_test = load_tfds_data(ds_test)
    data_dictionary_list = [{"train":images_train,  "test":images_test},
                            {"train":labels_train,  "test":labels_test},
                            {"train":labels_train[:,0,...], "test":labels_test[:,0, ...]},]
    return data_dictionary_list

def load_modelnet(selected_views, collection_list=None):
    if collection_list is None:
        collection_list = COMPLETE_COLLECTION
    data_parameters = {"data": "modelnet40",
                       "root_path": "/data/aligned",
                       "collection_list": collection_list,
                       "data_type": "train",
                       "dataset_directory": ""}

    dataset_class = load_factor_data(**data_parameters)
    dataset_class.images = dataset_class.images[:, :selected_views, ...]
    dataset_class.labels = dataset_class.labels[:, :selected_views, ...]
    print("Shape of dataset images train", dataset_class.images.shape)

    # Split the data for training, test and validation
    splitting_parameters = {"test_split": 0.0,
                            "val_split": 0.0,
                            "random_state": 0,
                            "stratify": True}

    # Split data for training per object
    data_dictionary_list = shrec_utils.train_test_val_split(
        [dataset_class.images, dataset_class.labels, dataset_class.labels[:, 0, ...]],
        **splitting_parameters)

    data_parameters["data_type"] = "test"
    dataset_class = load_factor_data(**data_parameters)
    dataset_class.images = dataset_class.images[:, :selected_views, ...]
    dataset_class.labels = dataset_class.labels[:, :selected_views, ...]
    print("Shape of dataset images test ", dataset_class.images.shape)
    data_dictionary_list[0]["test"] = dataset_class.images[:, :selected_views, ...]
    data_dictionary_list[1]["test"] = dataset_class.labels
    data_dictionary_list[2]["test"] = dataset_class.labels[:,0, ...]
    return data_dictionary_list


def load_data_dictionary_list(dataset, selected_views, test_split) -> Dict:
    """
    Return a list of dictionaries where the first element of the list are the images, the second the labels per image
    and the third the labels per object. Each dictionary has 3 keys: train, val, and test
    Args:
        dataset:
        selected_views:
        test_split:

    Returns:

    """
    if dataset == "shrec2021_shape":
        data_dictionary_list = load_shrec2021_challenge("Shape", selected_views, test_split)
    elif dataset == "shrec2021_culture":
        data_dictionary_list = load_shrec2021_challenge("Culture", selected_views, test_split)
    elif dataset == "modelnet40":
        data_dictionary_list = load_modelnet(selected_views)
    elif dataset == "cub_dataset":
        data_dictionary_list = load_cub_dataset()
    elif dataset == "cub_dataset_transforms":
        transforms = [segment_crop_norm_image, select_mapping_function(selected_views,"")]
        data_dictionary_list = load_tf_dataset_dictionary_transforms("caltech_birds2011", selected_views, transforms)
    elif dataset == "cub_dataset_colored_backgrounds":
        transforms = [select_mapping_colored_backgrounds(selected_views)]
        data_dictionary_list = load_tf_dataset_dictionary_transforms("caltech_birds2011", selected_views, transforms)
    elif dataset == "cars196":
        transforms = [crop_norm_image, select_mapping_function(selected_views,"")]
        data_dictionary_list = load_tf_dataset_dictionary_transforms("cars196", selected_views, transforms)
    else:
        data_dictionary_list = []
    assert len(
        data_dictionary_list) == 3, f"The data dictionary list does not have enough elements only {len(data_dictionary_list)}"

    print("Data dictionary training shape", data_dictionary_list[0]["train"].shape)
    return data_dictionary_list


class GroupedDataset:
    def __init__(self, dataset, selected_views, test_split):
        self.selected_views = selected_views
        self.test_split = test_split
        self.dataset_name = dataset
        if self.dataset_name == "cub_dataset_transforms":
            self.tf_dataset = True
        else:
            self.tf_dataset = False
        self.data_dictionary_list = load_data_dictionary_list(dataset, selected_views, test_split)


    @property
    def image_shape(self):
        return self.data_dictionary_list[0]["train"].shape[-3:]

    @property
    def num_object_dictionary(self):
        return shrec_utils.get_dictionary_object_quantities(self.data_dictionary_list[0])

    @property
    def flat_dictionary_list(self):
        return [shrec_utils.flatten_data_dictionary(data_dictionary) for data_dictionary in
                self.data_dictionary_list]

    @property
    def onehot_flat_dictionary(self):
        return shrec_utils.make_label_dictionary_onehot(self.flat_dictionary_list[1])

    @property
    def onehot_dictionary(self):
        return shrec_utils.make_label_dictionary_onehot(self.flat_dictionary_list[2])

    @property
    def class_weights_dictionary(self):
        return shrec_utils.get_dictionary_class_weights(self.data_dictionary_list[2])

    @property
    def flat_class_weights_dictionary(self):
        return shrec_utils.get_dictionary_class_weights(self.flat_dictionary_list[2])

    @property
    def transformations(self):
        num_images = len(self.data_dictionary_list[0]["train"])
        views = 2 * np.pi * np.linspace(0, 1, self.selected_views, endpoint=False)
        transformations_circular = np.expand_dims(np.array([views] * num_images), -1)
        transformations_euclidean = np.ones(transformations_circular.shape)
        transforms = [transformations_circular, transformations_euclidean]
        return transforms

    def get_class_weights_dictionary_model(self, model_type):
        if model_type in ["AE", "AEClass", "AETL", "VAE", "VAETL", "VAEClass", ]:
            class_weights = self.flat_class_weights_dictionary
        else:
            class_weights = self.class_weights_dictionary
        return class_weights

    def get_training_xy(self, model_type, batch_size = 200):
        if self.dataset_name == "cub_dataset_transforms":
            transforms = [segment_crop_norm_image, select_mapping_function(self.selected_views, model_type)]
            x = load_tensorflow_dataset_transforms("caltech_birds2011",  model_type, batch_size, transforms=transforms)[0]
            y = None
        if self.dataset_name == "cars196":
            transforms = [crop_norm_image, select_mapping_function(self.selected_views, model_type)]
            x = load_tensorflow_dataset_transforms("cars196", model_type, batch_size, transforms=transforms)[
                0]
            y = None

        else:
            if model_type == "AEClass" or model_type == "VAEClass":
                x = self.flat_dictionary_list[0]["train"]
                y = [self.flat_dictionary_list[0]["train"], self.onehot_dictionary["train"]]

            elif model_type == "GVAEClass":
                x = self.data_dictionary_list[0]["train"]
                one_hot_labels = shrec_utils.change_labels_to_onehot(self.data_dictionary_list[1]["train"][:, 0, ...])
                y = [self.data_dictionary_list[0]["train"], one_hot_labels]

            elif model_type == "AETL" or model_type == "VAETL":
                x = self.flat_dictionary_list[0]["train"]
                y = [self.flat_dictionary_list[0]["train"], self.flat_dictionary_list[1]["train"]]

            elif model_type == "GVAETL" or model_type == "GAETL":
                x = self.data_dictionary_list[0]["train"]
                y = [self.data_dictionary_list[0]["train"], self.data_dictionary_list[2]["train"].astype(float)]

            elif model_type == "LSBDVAETL":
                x = [self.data_dictionary_list[0]["train"], *self.transformations]
                y = [self.data_dictionary_list[0]["train"], self.data_dictionary_list[2]["train"]]

            elif model_type == "LSBDVAE":
                x = [self.data_dictionary_list[0]["train"], *self.transformations]
                y = self.data_dictionary_list[0]["train"]

            elif model_type == "TLtd":
                x = self.data_dictionary_list[0]["train"]
                y = self.data_dictionary_list[2]["train"]

            elif model_type == "GAE" or model_type == "GVAE" or model_type == "GAEProto":
                x = self.data_dictionary_list[0]["train"]
                y = None
            else:
                x = self.flat_dictionary_list[0]["train"]
                y = None

        return x, y

    def plot_example_images(self,object_num):
        fig, axes = plt.subplots(1, self.selected_views, figsize=(10, self.selected_views))
        for num_ax, ax in enumerate(axes):
            ax.imshow(self.data_dictionary_list[0]["train"][object_num, num_ax])
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, axes

    def plot_example_images_per_class(self):
        for label in np.unique(self.data_dictionary_list[1]["train"][:, 0]):
            fig, _ = self.plot_example_images(np.where(self.data_dictionary_list[2]["train"] == label)[0][0])
            plt.close(fig)
