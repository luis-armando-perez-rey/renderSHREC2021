import os
import imageio
import numpy as np
import h5py
import json
import argparse
import sys

PROJECT_PATH = os.path.dirname(os.getcwd())
sys.path.append(PROJECT_PATH)
from modules.utils.shrec_utils import obtain_models_classes_clafile
parser = argparse.ArgumentParser(
    description='Generate h5 files for data loading saves files by default in PROJECT_PATH/data/SHREC2021')
parser.add_argument('--resolution', nargs="+", dest='resolution', type=int, default=[256, 256],
                    help="resolution used to render images")
parser.add_argument('--nviews', nargs="?", dest='nviews', type=int, default=12,
                    help="number of views to generate per 3D model")
parser.add_argument('--render_path', nargs="?", dest='render_path', type=str, default="./renders",
                    help="input path where renders are")
parser.add_argument('--challenge', nargs="?", dest='challenge', type=str, default="Culture",
                      help="challenge from which to render either Culture or Shape")
parser.add_argument('--split', nargs="?", dest='split', type=str, default="test",
                      help="data to use for split either train or test")

args = parser.parse_args()

resolution = args.resolution
nviews = args.nviews

available_categories = ["Culture", "Shape"]
available_data_types = ["test", "train"]
# render_path = os.path.join(args.render_path, "dataset"+args.challenge, args.split+"_renders")
render_path = args.render_path
for selected_category in available_categories:
    for selected_data_type in available_data_types:

        dataset_path = os.path.join(render_path, "dataset" + selected_category)
        renders_path = os.path.join(dataset_path, selected_data_type + "_renders")

        # identifiers_path = os.path.join(dataset_path, selected_data_type + "_" + "object_dictionary")
        if selected_data_type== "train":
            categories, model_lists = obtain_models_classes_clafile(os.path.join(dataset_path, "dataset.cla"))
            identifiers = {}
            for num_class, model_list in enumerate(model_lists):
                for model_id in model_list:
                    identifiers.update({model_id: {"num_class": num_class, "class": categories[num_class]}})
            model_ids = identifiers.keys()
        else:
            model_ids = np.unique([file_name.split("_")[0] for file_name in os.listdir(renders_path)])


        h5_savepath = os.path.join(PROJECT_PATH, "data", "shrec2021")
        os.makedirs(h5_savepath, exist_ok=True)

        # with open(os.path.join(identifiers_path, 'data.json')) as json_file:
        #     identifiers = json.load(json_file)

        img_model_ids = np.array([int(image_filename.split("_")[0]) for image_filename in os.listdir(renders_path)])
        img_viewnum = np.array(
            [int(image_filename.split("_")[-1].split(".")[0]) for image_filename in os.listdir(renders_path)])



        # Initialize the array in which data is saved
        factors_shape = [len(model_ids), nviews]
        image_shape = [resolution[0], resolution[1], 3]
        images_shape = factors_shape + image_shape

        # Data
        images = np.zeros(images_shape, dtype=np.uint8)
        images_object_id = np.zeros(factors_shape, dtype=int)
        images_filenames = np.zeros(factors_shape, dtype="S10")
        images_classes_int = np.zeros(factors_shape, dtype=int)
        images_views_int = np.zeros(factors_shape, dtype=int)
        images_classes_str = np.zeros(factors_shape, dtype="S10")
        # images_object_dimensions = np.zeros(factors_shape + [3])

        # Loop through all the factors
        for num_id, model_id in enumerate(model_ids):
            print(type(model_id))
            print("Loading object {} with id {}".format(num_id, model_id))
            for view in range(nviews):
                print("Loading view number {} ".format(view))
                image_filename = str(model_id) + "_" + str(view) + ".png"

                # Store the read image into array
                images_filenames[num_id, view] = image_filename
                images[num_id, view, ...] = imageio.imread(os.path.join(renders_path, image_filename))[:, :, :3]
                images_views_int[num_id, view] = view
                images_object_id[num_id, view] = model_id
                if selected_data_type == "train":
                    images_classes_int[num_id, view] = identifiers[str(model_id)]["num_class"]
                    images_classes_str[num_id, view, ...] = identifiers[str(model_id)]["class"]
                # for dimension_num in range(3):
                    # images_object_dimensions[num_id, view, dimension_num] = identifiers[model_id][
                    #     "d" + str(dimension_num + 1)]

        # Save in an h5 file.
        dataset_filename = selected_category + "_" + selected_data_type + ".h5"
        print(dataset_filename)
        with h5py.File(os.path.join(h5_savepath, dataset_filename), 'w') as g:
            g.create_dataset('images', data=images, dtype=np.uint8)
            g.create_dataset('filenames', data=images_filenames, dtype="S10")
            # g.create_dataset('dimensions', data=images_object_dimensions, dtype=float)
            g.create_dataset('class_int', data=images_classes_int, dtype=int)
            g.create_dataset('class_str', data=images_classes_str, dtype="S10")
            g.create_dataset('views', data=images_views_int, dtype=int)
            g.create_dataset('ids', data=images_object_id, dtype=int)
