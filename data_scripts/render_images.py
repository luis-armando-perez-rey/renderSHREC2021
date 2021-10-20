import bpy
import os
import numpy as np
import glob
import time
import argparse
import sys

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--resolution', nargs="+", dest='resolution', type=int, default=[256, 256], help="resolution used to render images")
  parser.add_argument('--nviews', nargs="?", dest='nviews', type=int, default=12,
                      help="number of views to generate per 3D model")
  parser.add_argument('--input_path', nargs="?", dest='input_path', type=str, default="default",
                      help="input path where 3D models are")
  parser.add_argument('--output_path', nargs="?", dest='output_path', type=str, default="default",
                      help="output path where 3D models are placed")
  parser.add_argument('--max_obj_dim', nargs="?", dest='max_obj_dim', type=float, default=15.0,
                      help="maximum dimension in renders")
  parser.add_argument('--startmodel', nargs="?", dest='startmodel', type=int, default=0,
                      help="model number from which to start rendering")
  parser.add_argument('--challenges', nargs="+", dest='challenges', type=str, default=["Culture", "Shape"],
                      help="challenge from which to render either Culture or Shape")
  parser.add_argument('--splits', nargs="+", dest='splits', type=str, default=["test", "train"],
                      help="data to use for split either train or test")

  return parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])

arguments, unknown = parse_arguments()



start_time = time.time()


# Parse the arguments
resolution = arguments.resolution
max_object_dimension = arguments.max_obj_dim
nviews = arguments.nviews
start_model = arguments.startmodel

# Paths
if arguments.input_path == "default":
    input_path = DATA_PATH
else:
    input_path = arguments.input_path
if arguments.output_path == "default":
     output_path = os.path.join(DATA_PATH, "renders")
else:
    output_path = arguments.output_path

for challenge in arguments.challenges:
    for split in arguments.splits:
        object_folder = os.path.join(input_path, "dataset"+challenge, split)
        save_render_path = os.path.join(output_path, "dataset"+challenge, split+"_renders")
        os.makedirs(save_render_path, exist_ok=True)
        print(f"Rendering images of challenge {challenge} with split {split}")
        print("3D model folder", object_folder)
        print("Renders path", save_render_path)

        # Define rotation angles
        views = 2 * np.pi * np.linspace(0, 1, nviews, endpoint=False)

        # Set the context scene and camera
        context = bpy.context
        scene = context.scene
        camera = scene.camera

        # Set camera properties
        camera.location = [20.523, -19.961, 8.931]
        camera.rotation_mode = 'XYZ'
        camera.rotation_euler = [73.6 * np.pi / 180.0, 0.757 * np.pi / 180.0, 46.2 * np.pi / 180]
        camera.scale = [1.0, 1.0, 1.0]

        # Set rendering properties
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]

        #   Set world properties
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

        # OBJ files directory
        print(os.listdir(object_folder))
        list_paths = np.array(list(glob.glob(os.path.join(object_folder, "*.obj"))))
        print(os.path.join(object_folder, "*.obj"))
        object_id_list = np.array([int(os.path.basename(path).replace(".obj", "")) for path in list_paths])
        index_ordering = np.argsort(object_id_list)
        list_paths = list_paths[index_ordering]
        list_paths = list_paths[start_model:]
        object_id_list = object_id_list[index_ordering]
        object_id_list = object_id_list[start_model:]
        print("List of found paths", list_paths)
        for num_object, object_path in enumerate(list_paths):
            object_id = object_id_list[num_object]

            imported_object = bpy.ops.import_scene.obj(filepath=object_path)
            obj_object = bpy.context.selected_objects[0]  ####<--Fix
            print('Imported name: ', obj_object.name)

            scale_factor = max_object_dimension / max(obj_object.dimensions)
            obj_object.scale = np.ones(3) * scale_factor

            # Render several images
            for num_view, view in enumerate(views):
                print("Rendering view number" + str(num_view))
                obj_object.rotation_euler[2] = view

                filename = str(object_id) + "_" + str(num_view) + ".png"
                context.scene.render.filepath = os.path.join(save_render_path, filename)
                bpy.ops.render.render(write_still=True)

            meshes_to_remove = []
            for ob in bpy.context.selected_objects:
                meshes_to_remove.append(ob.data)
            bpy.ops.object.delete()
            # Remove the meshes from memory too
            for mesh in meshes_to_remove:
                bpy.data.meshes.remove(mesh)

        print("--- %s seconds ---" % (time.time() - start_time))


