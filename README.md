# SHREC2021 Retrieval of Cultural Heritage: Triplet Loss and Autoencoder 
Code used to render images for the Triplet Loss and Autoencoder algorithms used for the SHape Retrieval Challenge (SHREC) of 2021 on the Retrieval of Cultural Heritage Objects http://www.ivan-sipiran.com/shrec2021.html. This challenge consists on the retrieval of 3D models of pre-columbian peruvian artifacts based on two criteria: shape and culture. In this repository we describe the approach of Section 3.4 [1] that uses triplet loss and autoencoder methods.

# Requirements
Blender 2.82 https://www.blender.org/download/releases/2-82/

# Rendering
To render images from the 3D models first download the datasets from http://www.ivan-sipiran.com/shrec2021.html and unzip the datasets in /data. The resulting structure of the data folder should be:
```
renderSHREC2021
└---data
    └---datasetCulture
        └---train
            1.obj
            2.obj
            ...
        └---test
            1.obj
            2.obj
            ...
    └---datasetShape
        └---train
                1.obj
                2.obj
                ...
            └---test
                1.obj
                2.obj
                ...
```

In order to render the images for the Culture and Shape challenge after installation of blender run the following command from the main project folder:
```bash
blender -b ./data_scripts/render_shrec2021.blend -P ./data_scripts/render_images.py -- --input_path ./data --challenge Culture Shape --split train test
```
Then run the following command to create h5 files from the rendered images to load images more easily
```bash
python ./data_scripts/make_h5_file.py
```

# Docker container
To create a docker container run in the main project folder
```bash
docker build .
```
Run the command to get the id of the image
```bash
docker images
```
The output should look like
```
REPOSITORY         TAG       IMAGE ID       CREATED       SIZE
REPOSITORY_ID      TAG_ID    IMAGE_ID      10 sec ago     2.9GB
```


Finally run 
```bash
docker run --name Project v ./data:/data IMAGE_ID ./run_culture ./run 
```

# Reproducing results
To reproduce the results presented in [1] after having created the h5 files please go to /experiments folder and run 
```bash
bash final_submission
```
This will train the models with the corresponding hyperparameters for each challenge. The hyperparameters are described in detail in [1].

### References
[1] Sipiran, I., Lazo, P., Lopez, C., Jimenez, M., Bagewadi, N., Bustos, B., Dao, H., Gangisetty, S., Hanik, M., Ho-Thi, N-P., Holenderski, M. J., Jarnikov, D. S., Labrada, A., Lengauer, S., Licandro, R., Nguyen, D-H., Nguyen-Ho, T-L., Perez Rey, L. A., Pham, B-D., Pham, M-K. & 6 others, Preiner, R., Schreck, T., Trinh, Q-H., Tonnaer, L. M. A., von Tycowicz, C. & Vu-Le, T-A., "SHREC 2021: Retrieval of cultural heritage objects" (2021), In Computers and Graphics. 100, p. 1-20 20 p.






If you use this code please reference the following paper:
```
@article{SIPIRAN20211,
title = {SHREC 2021: Retrieval of cultural heritage objects},
journal = {Computers & Graphics},
volume = {100},
pages = {1-20},
year = {2021},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2021.07.010},
url = {https://www.sciencedirect.com/science/article/pii/S0097849321001412},
author = {Ivan Sipiran and Patrick Lazo and Cristian Lopez and Milagritos Jimenez and Nihar Bagewadi and Benjamin Bustos and Hieu Dao and Shankar Gangisetty and Martin Hanik and Ngoc-Phuong Ho-Thi and Mike Holenderski and Dmitri Jarnikov and Arniel Labrada and Stefan Lengauer and Roxane Licandro and Dinh-Huan Nguyen and Thang-Long Nguyen-Ho and Luis A. {Perez Rey} and Bang-Dang Pham and Minh-Khoi Pham and Reinhold Preiner and Tobias Schreck and Quoc-Huy Trinh and Loek Tonnaer and Christoph {von Tycowicz} and The-Anh Vu-Le},
keywords = {Benchmarking, 3D model retrieval, Cultural heritage},
abstract = {This paper presents the methods and results of the SHREC’21 track on a dataset of cultural heritage (CH) objects. We present a dataset of 938 scanned models that have varied geometry and artistic styles. For the competition, we propose two challenges: the retrieval-by-shape challenge and the retrieval-by-culture challenge. The former aims at evaluating the ability of retrieval methods to discriminate cultural heritage objects by overall shape. The latter focuses on assessing the effectiveness of retrieving objects from the same culture. Both challenges constitute a suitable scenario to evaluate modern shape retrieval methods in a CH domain. Ten groups participated in the challenges: thirty runs were submitted for the retrieval-by-shape task, and twenty-six runs were submitted for the retrieval-by-culture task. The results show a predominance of learning methods on image-based multi-view representations to characterize 3D objects. Nevertheless, the problem presented in our challenges is far from being solved. We also identify the potential paths for further improvements and give insights into the future directions of research.}
}
```
