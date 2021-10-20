import os
import h5py
import numpy as np
from data.factor_dataset import FactorImageDataset

AVAILABLE_COLLECTIONS = ["Shape", "Culture"]
AVAILABLE_DATA_TYPES = ["train", "test"]


def load_factor_data(data, root_path=None, **kwargs):
    options_dict = {
        "shrec2021": get_h5_saved_data,
    }
    return options_dict[data](root_path, **kwargs)


def get_h5_saved_data(root_path, collection_list, data_type, dataset_directory, normalize=True):
    """
    Returns a TransformImage object created from ModelNet40 dataset of objects with periodic colors and rotated
    Args:
        root_path: path to the root of the project
        dataset_filename: filename of the .h5 data to be loaded
        object_type: type of object saved in the data file
        normalize: whether data should be in the range [0,1] (True) or [0, 255] (False).

    Returns:
        FactorImageDataset object
        :param normalize:
        :param dataset_directory:
        :param data_type:
        :param collection_list:
    """
    image_list = []
    views_list = []
    labels_list = []
    ids_list = []
    for collection in collection_list:
        assert collection in AVAILABLE_COLLECTIONS, "collection_list = {} is not available. Possible values are {}".format(
            collection, AVAILABLE_COLLECTIONS)
        assert data_type in AVAILABLE_DATA_TYPES, "data_type = {} is not available. Possible values are {}".format(
            data_type, AVAILABLE_DATA_TYPES)

        dataset_filename = collection + "_" + data_type + ".h5"
        print(dataset_filename)
        dataset_filepath = os.path.join(root_path, dataset_directory, dataset_filename)
        # Read the images
        images = read_data_h5(dataset_filepath, "images")
        if normalize:
            images = images.astype('float32') / np.amax(images)
        image_list.append(images)
        # Read the rotations
        views = read_data_h5(dataset_filepath, "views")
        views_list.append(views)
        # Read category integer class_labels
        try:
            labels = read_data_h5(dataset_filepath, "class_int")
        except:
            print("No labels detected in the dataset")
            labels = None
        labels_list.append(labels)
        ids = read_data_h5(dataset_filepath, "ids")
        ids_list.append(ids)

    images = np.concatenate(image_list, axis=0)
    ids = np.concatenate(ids_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    views = np.concatenate(views_list, axis=0)
    # Convert integer range to angular range
    unique_views = np.unique(views)
    unique_ids = np.unique(ids)
    # Create FactorImageDataset lists
    factor_values = [unique_ids, unique_views]

    max_factor_values = [np.amax(factor) for factor in factor_values]
    return FactorImageDataset(images=images,
                              factor_values_list=factor_values,
                              max_factor_values=max_factor_values,
                              factor_names=["object_ids", "rotation_angle"],
                              labels=labels)


def read_data_h5(data_filepath, data_type):
    """
    Read data from h5 file with 1 level of hierarchy
    Args:
        data_filepath: path to the .h5 file
        data_type: key value from which data is read

    Returns:

    """
    with h5py.File(data_filepath, "r") as file:
        data = np.array(file.get(data_type))
    return data
