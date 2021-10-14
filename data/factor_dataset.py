from typing import Tuple, List
import numpy as np


class FactorImageDataset:
    """
    Abstract class for datasets that are a Cartesian product of factors

    Args:
        images: the image data itself, shape (f1, ..., fk, h, w, d)
        factor_values_list: list of 1D arrays, one for each factor, containing all factor values present in the data.
            If None, max_factor_values must be provided, and equal spacing is assumed based on these max values.
        max_factor_values: list of maximum values that indicate scale of factors (minimum is assumed to be 0).
            If None, the number of factors in images is used.
        factor_names: list of strings indicating the meaning of each factor.
            If None, ["factor0", ..., "factor{k-1}"] is used.

    Attributes:
        images (np.ndarray): the image data itself, shape (f1, ..., fk, h, w, d)
        factor_values_list: list of 1D arrays, one for each factor, containing all factor values present in the data
        max_factor_values: list of maximum values that indicate scale of factors (minimum is assumed to be 0)
        factor_names: list of strings indicating the meaning of each factor
    """

    def __init__(self, images: np.ndarray,
                 factor_values_list: List[np.ndarray] = None,
                 max_factor_values: List[float] = None,
                 factor_names: List[str] = None,
                 labels=None,
                 ):
        assert len(images.shape) >= 3, "images must have shape (..., h, w, d)"
        self.images = images

        # setup and/or check factor_values_list and max_factor_values
        if factor_values_list is None:  # if no factors are given, assume they are equally spaced
            if max_factor_values is None:  # if no max values given, assume n_factors as max
                max_factor_values = list(self.factors_shape)
            else:
                assert len(max_factor_values) == self.n_factors, \
                    "max_factor_values length should equal the number of factors"
            factor_values_list = []
            for max_factor_value, n_values in zip(max_factor_values, self.factors_shape):
                factor_values_list.append(np.linspace(0, max_factor_value, num=n_values, endpoint=False))
        else:
            assert max_factor_values is not None, "max_factor_values must be specified if factors are given"
            assert len(max_factor_values) == len(factor_values_list) == self.n_factors, \
                "max_factor_values and factor_values_list must have length equal to n_factors"

        # setup or check factor_names
        if factor_names is None:
            factor_names = [f"factor{i}" for i in range(self.n_factors)]
        else:
            assert len(factor_names) == self.n_factors, "factor_names length should equal the number of factors"

        # set attributes
        self.factor_values_list = factor_values_list
        self.max_factor_values = max_factor_values
        self.factor_names = factor_names

        # set class_labels
        self.labels = labels

    @property
    def factors_shape(self) -> Tuple:
        """
        Returns:
            factors_shape: data shape without the final (h, w, d) dimensions
        """
        return self.images.shape[:-3]

    @property
    def n_factors(self) -> int:
        """
        Returns:
            n_factors: the number of factors in images
        """
        return len(self.factors_shape)

    @property
    def factors_as_angles(self):
        """
        Returns:
            factors_as_angles: factor_values_list rescaled to a scale from 0 to 2pi
        """
        angles = []
        for factor, max_value in zip(self.factor_values_list, self.max_factor_values):
            angles.append(2 * np.pi * factor / max_value)
        return angles

    @property
    def image_shape(self):
        """
        Returns:
            image_shape: shape of the images (height, width, depth)
        """
        return self.images.shape[-3:]

    @property
    def n_data_points(self) -> int:
        """
        Returns:
            n_data_points: number of images in the full dataset
        """
        n_data_points = int(np.prod(self.factors_shape))
        return n_data_points

    @property
    def factor_mesh(self) -> np.ndarray:
        """
        Returns:
            factor mesh: array of shape (n1, n2, n3, ..., n_nfactors, nfactors)
        """
        factor_meshes = np.meshgrid(*self.factor_values_list, indexing="ij")
        factor_mesh = np.stack(factor_meshes, axis=-1)  # (n1, n2, n3, ..., n_n_factors ,n_factors)
        return factor_mesh

    @property
    def factor_mesh_enumerated(self) -> np.ndarray:
        """
        Returns:
            factor mesh: array of shape (n1, n2, n3, ..., n_nfactors, nfactors)
        """
        enumerated_factor_list = [np.arange(0, len(self.factor_values_list[num_factor])) for num_factor in
                                  range(self.n_factors)]
        factor_meshes = np.meshgrid(*enumerated_factor_list, indexing="ij")
        factor_mesh = np.stack(factor_meshes, axis=-1)  # (n1, n2, n3, ..., n_n_factors ,n_factors)
        return factor_mesh

    @property
    def flat_factor_mesh_enumerated(self) -> np.ndarray:
        """
        Returns:
            flat_factor_mesh: array of shape (n1*n2*n3...*n_nfactors, nfactors)
        """
        return self.factor_mesh_enumerated.reshape((self.n_data_points, self.n_factors))

    @property
    def factor_mesh_as_angles(self) -> np.ndarray:
        """
        Returns:
            factor_mesh_as_angles: same as factor_mesh, but with values scaled from 0 to 2pi
        """
        factor_meshes = np.meshgrid(*self.factors_as_angles, indexing="ij")
        factor_mesh = np.stack(factor_meshes, axis=-1)  # (n1, n2, n3, ..., n_n_factors ,n_factors)
        return factor_mesh

    @property
    def flat_factor_mesh(self) -> np.ndarray:
        """
        Returns:
            flat_factor_mesh: array of shape (n1*n2*n3...*n_nfactors, nfactors)
        """
        return self.factor_mesh.reshape((self.n_data_points, self.n_factors))

    @property
    def flat_factor_mesh_as_angles(self) -> np.ndarray:
        """
        Returns:
            flat_factor_mesh: same as flat_factor_mesh, but with values scaled from 0 to 2pi
        """
        return self.factor_mesh_as_angles.reshape((self.n_data_points, self.n_factors))

    @property
    def flat_images(self) -> np.ndarray:
        """
        Returns:
            flat_images: full dataset in the shape (n_data_points, h, w, d)
        """
        return self.images.reshape((self.n_data_points, *self.images.shape[-3:]))

    @property
    def flat_labels(self):
        """
        Returns:
            flat_images: full dataset in the shape (n_data_points, h, w, d)
        """
        if self.labels is None:
            print("No class_labels available")
            return self.labels
        else:
            return self.labels.reshape(self.n_data_points)

    def setup_circles_dataset_labelled_pairs(self, n_labels: int) -> (np.ndarray, List[np.ndarray], np.ndarray):
        """
        Args:
            n_labels: Number of labelled pairs to generate
        Returns:
            x_l: Labeled pairs array with shape (n_labels, 2, height, width, depth)
            x_l_transformations: List of length n_factors, each element is an array of shape (n_labels, 2, 1)
                where [:, 0, :] represents the identity transformations,
                and [:, 1, :] represents the transformation from the first to the second element of a pair,
                given as an angle on the unit circle
            x_u: Unlabeled data points with shape (n_data_points - 2*n_labels, 1, height, width, depth)
        """
        # labelling procedure: randomly select n_labels pairs, such that each data point is part of at most one pair.
        # produce the transformation label for each of those pairs.
        assert 2 * n_labels <= self.n_data_points, \
            "for this procedure 2 * n_labels cannot exceed the number of data points"
        flat_factor_mesh = self.flat_factor_mesh
        # sample 2*n_labels indices, for the data points/pairs to be labelled
        indices = np.random.choice(self.n_data_points, size=2 * n_labels, replace=False)
        # split in two halves, for the first and second elements of the pairs
        ind1 = indices[:n_labels]
        ind2 = indices[n_labels:]

        x_l_transformations = []
        for factor_num in range(self.n_factors):
            if self.factors_shape[factor_num] != 1:
                differences = (flat_factor_mesh[ind2, factor_num] - flat_factor_mesh[ind1, factor_num]) % \
                              self.max_factor_values[factor_num]
                angles = np.expand_dims(2 * np.pi * differences / self.max_factor_values[factor_num], axis=1)
                identity_transformations = np.zeros_like(angles)
                x_l_transformations.append(np.stack([identity_transformations, angles], axis=1))

        # set up the set x_l of labelled data points, with shape (n_labels, 2, height, width, depth)
        images_flat = self.flat_images
        x1 = images_flat[ind1]  # shape (n_labels, height, width, depth)
        x2 = images_flat[ind2]  # shape (n_labels, height, width, depth)
        x_l = np.stack([x1, x2], axis=1)  # shape (n_labels, 2, height, width, depth)

        # select all remaining data points for the unlabelled set x_u,
        #   with shape (n_unlabelled, 1, height, width, depth)
        mask = np.ones(self.n_data_points, dtype=bool)
        mask[indices] = False
        x_u = images_flat[mask]
        x_u = np.expand_dims(x_u, axis=1)  # shape (n_data_points - 2*n_labels, 1, height, width, depth)

        return x_l, x_l_transformations, x_u
