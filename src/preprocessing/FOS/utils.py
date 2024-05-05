import numpy as np
import pandas as pd
from distython import HEOM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset


class FOS_SMOTE:
    def __init__(self, k: int, random_state: int):
        self.k = k
        self.random_state = random_state

    def generate_examples(self, base: pd.DataFrame, neighbors: pd.DataFrame, dataset: Dataset, target_value: int):
        """first compute k-nearest neighbors of each instance in the base, then generate new instance
        :param target_value: value of the target for the synthetic examples generated
        :param base: base samples (as in the paper)
        :param neighbors: set of instances from which the neighbors will be computed, they should include the base instances
        :param dataset: dataset structure"""
        new_examples = []
        X_base, y_base = base.loc[:, base.columns != dataset.target], base[dataset.target]
        X_neighbors, y_neighbors = neighbors.loc[:, neighbors.columns != dataset.target], neighbors[dataset.target]
        cat_ord_features = [f for f, t in dataset.feature_types.items() if
                            (t == 'ordinal' or t == 'categorical') and f != dataset.target]
        cat_ord_features = [X_base.columns.get_loc(c) for c in cat_ord_features]
        X_base_instances = pd.concat([X_neighbors, X_base], axis=0)
        metric = HEOM(X_base_instances, cat_ord_features, nan_equivalents=[np.nan])
        knn = NearestNeighbors(n_neighbors=self.k + 1, metric=metric.heom, n_jobs=-1)
        knn.fit(X_neighbors)
        for idx in X_base.index:
            b = X_base.loc[[idx], :]
            distances, nearest_neighbors = knn.kneighbors(b)
            distances = distances.flatten()
            nearest_neighbors = nearest_neighbors.flatten()
            nearest_neighbors = np.array([X_base_instances.index[n] for n, d in zip(nearest_neighbors, distances) if d > 0])
            assert len(nearest_neighbors) == self.k, (distances, b)
            new_example = self._generate_synthetic_example(b, X_neighbors, nearest_neighbors, dataset)
            new_example[dataset.target] = target_value
            new_examples.append(new_example)
        new_examples = pd.concat(new_examples)
        return new_examples

    def _generate_synthetic_example(self, base: pd.DataFrame, neighbors: pd.DataFrame,
                                    nearest_neighbors: np.ndarray | list, dataset: Dataset) -> pd.DataFrame:
        """generates synthetic example as in SMOTE
        :param nearest_neighbors: indices of nearest neighbors
        :param base: the original instance
        :param neighbors: all set of potential neighbors
        :param dataset: the dataset structure
        :return: new synthetic example"""
        features_names = base.columns
        features = dataset.feature_types
        synthetic_example = {}
        nearest = neighbors.loc[nearest_neighbors, :]
        random_neighbor = dataset.random_state.choice(nearest_neighbors, size=1)[0]
        random_neighbor = neighbors.loc[[random_neighbor], :]

        for feature in features_names:
            x1_value = base[feature].to_numpy()[0]
            x2_value = random_neighbor[feature].to_numpy()[0]

            if features[feature] == 'continuous':
                dif = x1_value - x2_value
                gap = dataset.random_state.random()
                synthetic_example_value = x1_value - gap * dif

            elif features[feature] == 'ordinal':
                synthetic_example_value_float = (x1_value + x2_value) / 2
                synthetic_example_value = int(synthetic_example_value_float)

            elif features[feature] == 'categorical':
                # most common value
                all_datapoints = pd.concat([nearest, base], ignore_index=True)
                synthetic_example_value = all_datapoints.mode()[feature][0]
            else:
                exit("Feature type not valid: " + features[feature])

            synthetic_example[feature] = [synthetic_example_value]
        return pd.DataFrame(synthetic_example)
