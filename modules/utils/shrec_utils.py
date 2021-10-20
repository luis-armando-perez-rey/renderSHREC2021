import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns


# -----------------------------------------------
# Data utils
# -----------------------------------------------


def train_test_val_split(data, test_split, val_split, random_state, stratify=True):
    """
    Divides the available data into a dictionary of train test and val respectively. If no test or validation
    percentage is selected then no dictionary entry is created.
    Args:
        data:
        test_split:
        val_split:
        random_state:
        stratify:

    Returns:

    """
    if test_split != 0:
        if stratify:
            train_test = train_test_split(*data, stratify=data[1], test_size=test_split, random_state=random_state)
        else:
            train_test = train_test_split(*data, test_size=test_split, random_state=random_state)
        train = [train_test[num] for num in range(0, len(train_test), 2)]
        test = [train_test[num] for num in range(1, len(train_test), 2)]
    else:
        train = data
        test = []

    if val_split != 0:
        if stratify:
            train_val = train_test_split(*train, stratify=train[1], test_size=val_split, random_state=random_state)
        else:
            train_val = train_test_split(*train, test_size=val_split, random_state=random_state)
        train = [train_val[num] for num in range(0, len(train_val), 2)]
        val = [train_val[num] for num in range(1, len(train_val), 2)]
    else:
        train = train
        val = []

    data_dictionary_list = []
    for num_element in range(len(train)):
        data_dictionary = {}

        data_dictionary.update({"train": train[num_element]})
        if len(test) != 0:
            data_dictionary.update({"test": test[num_element]})
        else:
            print("No test data selected")

        if len(val) != 0:
            data_dictionary.update({"val": val[num_element]})
        else:
            print("No validation data selected")
        data_dictionary_list.append(data_dictionary)

    return data_dictionary_list


def get_dictionary_object_quantities(data_dictionary):
    num_object_dictionary = {}
    for key in data_dictionary.keys():
        num_object_dictionary.update({key: len(data_dictionary[key])})
    return num_object_dictionary


def flatten_data_dictionary(data_dictionary):
    flat_data_dictionary = {}
    for key in data_dictionary.keys():
        if len(data_dictionary[key]) != 0:
            if data_dictionary[key].ndim < 2:
                flat_data_dictionary.update({key: data_dictionary[key]})
            else:
                flat_data_dictionary.update({key: shrec_flatten_array(data_dictionary[key])})
        else:
            flat_data_dictionary.update({key: data_dictionary[key]})
    return flat_data_dictionary


def change_labels_to_onehot(labels):
    onehotencoder = OneHotEncoder()
    onehotencoder.fit(labels.reshape([-1, 1]))
    return onehotencoder.transform(labels.reshape([-1, 1])).toarray()


def make_label_dictionary_onehot(data_dictionary):
    onehot_dictionary = {}
    for key in data_dictionary.keys():
        if len(data_dictionary[key]) != 0:

            onehot_dictionary.update({key: change_labels_to_onehot(data_dictionary[key])})

        else:
            print(key, "has no class_labels")
            onehot_dictionary.update({key: np.array([])})
    return onehot_dictionary


def shrec_flatten_array(data, num_axes=2):
    """
    Flattens data input first num_axes
    Args:
        data: data array to be flattened
        num_axes: first num_axes to be flattened

    Returns:

    """
    return data.reshape((np.product([data.shape[ax] for ax in range(num_axes)]), *data.shape[num_axes:]))


def get_mask_dict(data_dictionary, key_val=None):
    """
    Create mask dict based on key value
    Args:
        data_dictionary:
        key_val:

    Returns:

    """
    if key_val is None:
        mask_dict = None
    else:
        mask_dict = {}
        for key in data_dictionary.keys():
            if len(data_dictionary[key]) != 0:
                mask_dict.update({key: data_dictionary[key] == key_val})
    return mask_dict


def mask_dictionary(data_dictionary, mask_dict=None):
    """
    Takes a data dictionary and filters out the arrays within according to the mask dictionary
    Args:
        data_dictionary: 
        mask_dict: 
    Returns:

    """
    if mask_dict is None:
        masked_dictionary = data_dictionary
    else:
        masked_dictionary = {}
        for key in data_dictionary.keys():
            if len(data_dictionary[key]) != 0 and len(mask_dict[key]) != 0:
                masked_dictionary.update({key: data_dictionary[key][mask_dict[key]]})
            else:
                masked_dictionary.update({key: []})
    return masked_dictionary


def create_views_dictionary(dictionary_, num_views):
    """
    Creates a dictionary with the views per object
    Args:
        dictionary_: dictionary with images per object for each data type (num_objects, ...)
        num_views: number of views per object

    Returns:

    """
    views_dictionary = {}
    for key in dictionary_.keys():
        if len(dictionary_[key]) != 0:
            views_dictionary.update({key: np.stack([np.arange(0, num_views)] * len(dictionary_[key]))})
        else:
            views_dictionary.update({key: []})
    return views_dictionary




def obtain_models_classes_clafile(cla_file_path):
    """
    Takes the path to a .cla file and returns two lists. The returned categories list corresponds to a list of
    strings that contains the class values. The second list contains a set of arrays that contains the number of the
    models in the corresponding class.
    :param cla_file_path:
    :return: categories, model_lists
    """
    with open(cla_file_path) as f:
        arr = [line.replace("\n", "") for line in f]
    # Read from the .cla file the number of categories
    num_categories = int(arr[1].split(" ")[0])
    initial_index = 3# Index for starting to read in the .cla file
    categories = []
    model_lists = []
    for num_category in range(num_categories):
        # Add a class to the category list
        categories.append(arr[initial_index].split(" ")[0])
        # Read in the .cla file the number of models for that given class
        print('Initial index',arr[initial_index])
        num_models = int(arr[initial_index].split(" ")[-1])
        # Read the model numbers from the .cla file and save them in an array
        array_models = []
        for num_model in range(num_models):
            array_models.append(arr[initial_index + 1 + num_model])
        initial_index += num_models + 2
        model_lists.append(array_models)
    return categories, model_lists



# -----------------------------------------------
# Plotting utils
# -----------------------------------------------

def get_dictionary_class_weights(label_dictionary):
    dictionary_weights = {}
    for key in label_dictionary.keys():
        dictionary_weights.update({key: np.ones(len(label_dictionary[key]))})
        for num_label, label in enumerate(np.unique(label_dictionary[key])):
            dictionary_weights[key][label_dictionary[key] == label] = \
                compute_class_weight("balanced", np.unique(label_dictionary[key]), label_dictionary[key])[num_label]
    return dictionary_weights


def get_embeddings_per_object(embeddings, num_objects, num_views):
    embeddings_dimensions = embeddings.shape[-1]
    reshaped_embeddings = embeddings.reshape([num_objects, num_views, embeddings_dimensions])
    return np.mean(reshaped_embeddings, axis=1)


def get_embeddings_per_object_dictionary(dictionary_embeddings, dictionary_objects, num_views):
    dictionary_embeddings_objects = {}
    for key in dictionary_embeddings.keys():
        if len(dictionary_embeddings[key]) != 0:
            dictionary_embeddings_objects.update(
                {key: get_embeddings_per_object(dictionary_embeddings[key], dictionary_objects[key], num_views)})
        else:
            dictionary_embeddings_objects.update({key: []})
    return dictionary_embeddings_objects


def get_labels_per_object(labels, num_objects, num_views):
    labels_reshaped = labels.reshape([num_objects, num_views])
    return labels_reshaped[:, 0]


def get_labels_per_object_dictionary(labels, dictionary_objects, num_views):
    dictionary_embeddings_objects = {}
    for key in labels.keys():
        if len(labels[key]) != 0:
            dictionary_embeddings_objects.update(
                {key: get_labels_per_object(labels[key], dictionary_objects[key], num_views)})
        else:
            dictionary_embeddings_objects.update({key: []})
    return dictionary_embeddings_objects


def apply_function_to_dictionary(prediction_function, dictionary):
    prediction_dictionary = {}
    for key in dictionary.keys():
        if len(dictionary[key]) != 0:
            prediction = prediction_function(dictionary[key])
            prediction_dictionary.update({key: prediction})
        else:
            print(key, "is empty")
            prediction_dictionary.update({key: []})
    return prediction_dictionary


def make_prediction_dictionary(model, dictionary):
    prediction_dictionary = {}
    print("Making prediction dictionary")
    for key in dictionary.keys():
        if len(dictionary[key]) != 0:
            prediction = model.predict(dictionary[key])
            if isinstance(prediction, list):
                prediction_dictionary.update({key: prediction[0]})
            else:
                prediction_dictionary.update({key: prediction})
        else:
            print(key, "is empty")
            prediction_dictionary.update({key: []})
    return prediction_dictionary


def join_dictionaries(list_dictionaries):
    joined_dictionary = list_dictionaries[0]
    for dictionary_data in list_dictionaries[1:]:
        for key in dictionary_data.keys():
            if len(dictionary_data[key]) != 0:
                if len(joined_dictionary[key]) != 0:
                    joined_dictionary[key] = np.concatenate((joined_dictionary[key], dictionary_data[key]), axis=0)
                else:
                    joined_dictionary[key] = dictionary_data[key]
    return joined_dictionary


def predict_parameters_lsbd(model, data):
    parameters = model.predict(data)
    print("Output parameters shapes", parameters[1].shape, parameters[3].shape)
    embeddings = np.concatenate((parameters[1], parameters[3]), axis=-1)
    return embeddings


def predict_parameters_lsbd_dictionary(model, data_dictionary):
    parameters_dictionary = {}
    for key in data_dictionary.keys():
        if len(data_dictionary[key]) != 0:
            parameters_dictionary.update({key: predict_parameters_lsbd(model, data_dictionary[key])})
        else:
            parameters_dictionary.update({key: []})
    return parameters_dictionary


def embeddings_object_lsbd_dictionary(data_dictionary):
    embeddings = {}
    for key in data_dictionary.keys():
        if len(data_dictionary[key]) != 0:
            # Select embeddings from the Euclidean latent space
            embeddings.update({key: np.mean(data_dictionary[key][:, :, 2:], axis=1)})
        else:
            embeddings.update({key: []})

    return embeddings


# -----------------------------------------------
# Plotting utils
# -----------------------------------------------
def plot_all_objects(images: np.ndarray,
                     total_columns: int = 10):
    """
    Plot an example image from all objects.
    Args:
        images: images organized into arrays with shape (n_objects, n_views, *image_shape)
        total_columns: total columns to use for the visualization

    Returns:

    """
    total_objects = len(images)
    total_rows = int(np.ceil(total_objects / total_columns))  # rows are calculated with respect to number of objects
    fig = plt.figure(figsize=(total_columns, total_rows))
    for i in range(total_objects):
        ax = fig.add_subplot(total_rows, total_columns, i + 1)
        ax.imshow(images[i, 0])
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def image_scatter(x, y, image, ax=None, zoom=1.0):
    if ax is None:
        ax = plt.gca()

    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def plot_objects_pca(object_images, embeddings, zoom=1.0, ax=None):
    """
    Plot images on the same location as the embeddings with certain zoom
    Args:
        object_images: images of the objects to be plotted
        zoom: zoom factor for the plotted images
        embeddings: embeddings to be projected by PCA

    Returns:

    """
    # Plot embeddings
    if embeddings.shape[-1] == 2:
        print("PCA: Embeddings are already 2 dimensional, no PCA applied")
        x_embedded = embeddings
    else:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        x_embedded = pca.transform(embeddings)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = plt.gcf()
    for num_embedding, embedding in enumerate(x_embedded):
        image_scatter(embedding[0], embedding[1], object_images[num_embedding], ax=ax, zoom=zoom)
    ax.set_title("PCA object embeddings")
    return fig, ax


def plot_embeddings_pca(embeddings, colors=None, ax=None, alpha=1.0):
    """
    Plot embeddings
    Args:
        embeddings:
        colors:

    Returns:

    """
    # Plot embeddings
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    x_embedded = pca.transform(embeddings)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()
    if colors is not None:
        ax.scatter(x_embedded[:, 0], x_embedded[:, 1], c=colors, alpha=alpha)
    else:
        ax.scatter(x_embedded[:, 0], x_embedded[:, 1], alpha=alpha)
    ax.legend()
    ax.set_title("PCA embeddings")
    return fig, ax


def plot_example_object_per_class(images, labels, num_view=0):
    """
    Plots examples of the images available for each class
    Args:
        images: flat image array (num_examples, *image_dimensions)
        labels: flat label array (num_examples)
        num_view: integer

    Returns:

    """
    num_classes = len(np.unique(labels))
    fig, axes = plt.subplots(1, num_classes, figsize=(10, 5))
    for num_ax, ax in enumerate(axes):
        ax.imshow(images[labels == num_ax][num_view])
        ax.set_title(str(labels[labels == num_ax][num_view]))
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def plot_confusion_matrix(y_test, y_prediction):
    cm = confusion_matrix(y_test, y_prediction)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm / np.sum(cm), annot=True,
                fmt='.1%')
    return fig


def plot_reconstructions(images, reconstructions, num_object, num_views):
    """
    Plot images and reconstructions which are ordered as (num_object, num_view, *image_dimensions)
    Args:
        images:
        reconstructions:
        num_object:
        num_views: number of views to show

    Returns:

    """

    assert images.shape == reconstructions.shape, (f"Images shape {images.shape} is not the same as reconstructions"
                                                   f" shape {reconstructions.shape}")
    fig, axes = plt.subplots(2, num_views, figsize=(15, 3))

    if num_views == 1:
        axes[0].set_title("")
        axes[0].imshow(images[num_object, 0])
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(reconstructions[num_object, 0])

        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[0].set_ylabel("Original")
        axes[1].set_ylabel("Reconstruction")
    else:
        for num_view in range(num_views):
            axes[0, num_view].set_title("")
            axes[0, num_view].imshow(images[num_object, num_view])
            axes[0, num_view].set_xticks([])
            axes[0, num_view].set_yticks([])

            axes[1, num_view].imshow(reconstructions[num_object, num_view])

            axes[1, num_view].set_xticks([])
            axes[1, num_view].set_yticks([])

            if num_view == 0:
                axes[0, num_view].set_ylabel("Original")
                axes[1, num_view].set_ylabel("Reconstruction")
    return fig, axes


def plot_reconstructions2(images, reconstructions, num_object, num_views, num_columns=10):
    """
    Plot images and reconstructions which are ordered as (num_object, num_view, *image_dimensions)
    Args:
        images:
        reconstructions:
        num_object:
        num_views: number of views to show

    Returns:

    """

    assert images.shape == reconstructions.shape, (f"Images shape {images.shape} is not the same as reconstructions"
                                                   f" shape {reconstructions.shape}")
    total_rows = int(2 * np.ceil(num_views / num_columns))  # rows are calculated with respect to number of objects
    # fig, axes = plt.subplots(total_rows, num_columns, figsize=(num_columns, total_rows))
    fig = plt.figure(figsize=(num_columns, total_rows))

    for i in range(num_views):
        # Plot original image
        ax = fig.add_subplot(total_rows, num_columns, 2 * i + 1)
        ax.imshow(images[num_object, i])
        ax.set_xticks([])
        ax.set_yticks([])
        if i < num_columns // 2:
            ax.set_title("Original")

        ax = fig.add_subplot(total_rows, num_columns, 2 * i + 2)
        ax.imshow(reconstructions[num_object, i])
        ax.set_xticks([])
        ax.set_yticks([])
        if i < num_columns // 2:
            ax.set_title("Rec")

    return fig


def plot_reconstructions_per_class(flat_images, reconstructions, class_labels):
    num_classes = len(np.unique(class_labels))
    fig, axes = plt.subplots(2, num_classes, figsize=(10, 3))

    for num_class in range(num_classes):
        axes[0, num_class].set_title(str(class_labels[class_labels == num_class][0]))
        axes[0, num_class].imshow(flat_images[class_labels == num_class][0])
        axes[0, num_class].set_xticks([])
        axes[0, num_class].set_yticks([])

        axes[1, num_class].imshow(reconstructions[class_labels == num_class][0])

        axes[1, num_class].set_xticks([])
        axes[1, num_class].set_yticks([])

        if num_class == 0:
            axes[0, num_class].set_ylabel("Original")
            axes[1, num_class].set_ylabel("Reconstruction")
    return fig, axes


def fig_pca_dictionary(embeddings_dictionary, labels_dictionary):
    """
    Creates a dictionary of figures showing pca projections with each key showing the figure formed
    by appliyng pca to embeddings from a dictionary and colored according to class_labels from another dictionary.

    Args:
        embeddings_dictionary: dictionary with embeddings
        labels_dictionary: dictionary with class_labels

    Returns:

    """
    assert set(embeddings_dictionary.keys()) == set(
        labels_dictionary.keys()), "Embeddings {} and class_labels {} don't have the same keys ".format(
        embeddings_dictionary.keys(), labels_dictionary.keys())
    fig_dictionary = {}
    for key in embeddings_dictionary.keys():
        if len(embeddings_dictionary[key]) != 0:
            fig, ax = plot_embeddings_pca(embeddings_dictionary[key], labels_dictionary[key])
            fig_dictionary.update({key: fig})
    return fig_dictionary


