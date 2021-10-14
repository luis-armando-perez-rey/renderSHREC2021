import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(matrix, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("Query")
    ax.set_xlabel("Collection")
    # plt.colorbar()
    return fig, ax


def l2_distance(x, y):
    """
    Calculates euclidean distance between two embeddings
    Args:
        x:
        y:

    Returns:

    """
    return np.linalg.norm(x - y, axis=-1)


def l1_distance(x, y):
    """
        Calculates euclidean distance between two embeddings
        Args:
            x:
            y:

        Returns:

        """
    return np.linalg.norm(x - y, ord="1", axis=-1)


def cosine_distance(x, y):
    """
    Cosine similarity between two embeddings
    Args:
        x:
        y:

    Returns:

    """
    return 1 - (np.sum(x * y, axis=-1) / (
                np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + np.finfo(float).eps))


class ShrecEmbeddingDistance:
    def __init__(self, query_embeddings, collection_embeddings, distance_function):
        self.query_embeddings = query_embeddings
        self.collection_embeddings = collection_embeddings
        self.distance_function = distance_function
        self.distance_matrix = self.set_distance_matrix()
        self.ranked_indexes = self.set_ranked_indexes_collection()

    @property
    def num_queries(self):
        return len(self.query_embeddings)

    @property
    def num_collection_elements(self):
        return len(self.collection_embeddings)

    def set_distance_matrix(self, duplicate_query=True):
        """
        Creates a distance matrix between query embeddings and collection embeddings
        Returns:

        """
        distance_matrix = np.zeros((self.num_queries, self.num_collection_elements))
        # Calculate faster distance matrix if distance function can be calculated in batches
        if duplicate_query:
            for num_query, query_embedding in enumerate(self.query_embeddings):
                # print("Calculating distance num query {}".format(num_query))
                duplicate_query = self.duplicate_embedding(query_embedding, self.num_collection_elements)
                distance_matrix[num_query, :] = self.distance_function(duplicate_query, self.collection_embeddings)
        else:
            for num_query, query_embedding in enumerate(self.query_embeddings):
                for num_collection, collection_embedding in enumerate(self.collection_embeddings):
                    distance_matrix[num_query, num_collection] = self.distance_function(query_embedding,
                                                                                        collection_embedding)
        return distance_matrix

    def duplicate_embedding(self, embedding, times):
        assert embedding.ndim == 1, "Embedding number of dimensions is not 1, it is {}".format(embedding.ndim)
        return np.array([list(embedding)] * times)

    def set_ranked_indexes_collection(self):
        """
        Creates a matrix of indexes to order collection embeddings per row according to distance
        Returns:

        """
        return self.distance_matrix.argsort(axis=-1)

    def save_submission(self, filepath):
        """
        Save the distance matrix for submission to SHREC
        Args:
            filepath: complete path for saving results

        Returns:

        """
        with open(filepath, "w") as output:
            for num_query in range(self.num_queries):
                for num_collection in range(self.num_collection_elements):
                    output.write(str(self.distance_matrix[num_query, num_collection]) + " ")
                output.write("\n")


class ShrecEvaluation(ShrecEmbeddingDistance):
    def __init__(self, query_embeddings, collection_embeddings, distance_function, query_labels, collection_labels):
        self.query_labels = query_labels
        self.collection_labels = collection_labels
        ShrecEmbeddingDistance.__init__(self, query_embeddings, collection_embeddings, distance_function)
        self.num_relevant_per_query = self.set_num_relevant_per_query()
        self.true_positive_matrix = self.set_true_positive_matrix()
        self.precision_matrix = self.set_precision_matrix()
        self.recall_matrix = self.set_recall_matrix()
        self.first_tier, self.second_tier = self.set_first_second_tier()

    def set_true_positive_matrix(self):
        """
        Calculate true positives between query and collection


        Returns:

        """
        assert len(self.query_labels) == self.num_queries, "{} query class_labels provided, {} are needed".format(
            len(self.query_labels), self.num_queries)
        assert len(
            self.collection_labels) == self.num_collection_elements, "{} collection class_labels provided, {} are needed".format(
            len(self.collection_labels), self.num_collection_elements)
        true_positive_matrix = np.zeros(self.distance_matrix.shape, dtype=bool)
        for num_query, query_label in enumerate(self.query_labels):
            true_positive_matrix[num_query] = (query_label == self.collection_labels[self.ranked_indexes[num_query]])
        return true_positive_matrix

    @property
    def mean_average_precision(self):
        return np.sum(self.true_positive_matrix * self.precision_matrix, axis=-1) / self.num_relevant_per_query

    @property
    def nn_precision(self):
        return np.mean(self.precision_matrix[:, 0])

    @property
    def prediction(self):
        """
        Per query return the class of the nearest neighbor in the collection set
        Returns:

        """
        return self.collection_labels[list(self.ranked_indexes[:, 0])]

    @property
    def metric_dictionary(self):
        return {"MAP": np.mean(self.mean_average_precision),
                "nn": self.nn_precision,
                "first_tier": np.mean(self.first_tier),
                "second_tier": np.mean(self.first_tier)}

    def set_precision_matrix(self):
        pm = np.zeros(self.true_positive_matrix.shape, dtype=float)
        for num_collection in range(self.num_collection_elements):
            pm[:, num_collection] = np.sum(self.true_positive_matrix[:, :(num_collection + 1)],
                                           axis=1) / (num_collection + 1)
        return pm

    def set_num_relevant_per_query(self):
        return np.array([np.sum(query_label == self.collection_labels) for query_label in self.query_labels])

    def set_first_second_tier(self):
        st = np.zeros(self.num_queries)
        ft = np.zeros(self.num_queries)
        for num_query in range(self.num_queries):
            second_tier_index = np.amin(
                [(2 * self.num_relevant_per_query[num_query]) - 1, self.num_collection_elements - 1])
            ft[num_query] = self.recall_matrix[num_query, self.num_relevant_per_query[num_query] - 1]
            st[num_query] = self.recall_matrix[num_query, second_tier_index]
        return ft, st

    def set_recall_matrix(self):
        rm = np.zeros(self.true_positive_matrix.shape, dtype=float)

        for num_collection in range(self.num_collection_elements):
            rm[:, num_collection] = np.sum(self.true_positive_matrix[:, :num_collection],
                                           axis=1) / self.num_relevant_per_query
        return rm

    def plot_precision_recall_num_query(self, num_query):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.recall_matrix[num_query], self.precision_matrix[num_query])
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        ax.set_title("Precision vs Recall Plot")
        ax.grid()
        return fig, ax
