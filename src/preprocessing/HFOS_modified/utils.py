from copy import deepcopy

import numpy as np
import pandas as pd
from distython import HEOM
# from distython import HEOM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset, query_dataset


class HFOS_SMOTE:
    def __init__(self, k: int, knn):
        self.k = k
        self.knn = knn

    def generate_example(self, instance: pd.DataFrame, cluster: pd.DataFrame, dataset: Dataset):
        new_example = {}

        X_instance, y_instance = instance.loc[:, instance.columns != dataset.target], instance[dataset.target]
        X_cluster, y_cluster = cluster.loc[:, instance.columns != dataset.target], cluster[dataset.target]

        neighbors = self.compute_random_neighbor(X_instance, X_cluster, dataset)
        neighbor = dataset.random_state.choice(neighbors, size=1)[0]
        neighbor = cluster.loc[[neighbor], :]
        cluster_neighbors = cluster.loc[neighbors, :]

        X_neighbor, y_neighbor = neighbor.loc[:, neighbor.columns != dataset.target], neighbor[dataset.target]

        knns, dist = self.compute_nearest_neighbors_whole(X_instance, y_instance, dataset)
        weight = self.compute_weight(knns, y_instance)
        weight = dataset.random_state.uniform(low=0, high=weight, size=1)[0]
        distance_factor = self.compute_distance_factor(X_instance, X_neighbor, dist)

        features = dataset.feature_types
        for feature in X_instance.columns:
            x1_value = instance[feature].to_numpy()[0]
            x2_value = neighbor[feature].to_numpy()[0]

            if features[feature] == 'continuous':
                # dif = x1_value - x2_value
                # gap = dataset.random_state.random()
                synthetic_example_value = x1_value + weight * (x2_value - x1_value) * distance_factor

            elif features[feature] == 'ordinal':
                synthetic_example_value_float = (x1_value + x2_value) / 2
                synthetic_example_value = int(synthetic_example_value_float)

            elif features[feature] == 'categorical':
                # most common value
                all_datapoints = pd.concat([cluster_neighbors, instance], ignore_index=True)
                synthetic_example_value = all_datapoints.mode()[feature][0]
            else:
                exit("Feature type not valid: " + features[feature])

            new_example[feature] = [synthetic_example_value]
            new_example[dataset.target] = [y_instance.to_numpy()[0]]
        return pd.DataFrame(new_example)

    def compute_nearest_neighbors_whole(self, X_instance: pd.DataFrame, y_instance: pd.DataFrame, dataset: Dataset):
        X_train, y_train = dataset.features_and_classes("train")

        distances, nearest_neighbors = self.knn.kneighbors(X_instance)
        distances = distances.flatten()
        nearest_neighbors = nearest_neighbors.flatten()
        distances = [d for n, d in zip(nearest_neighbors, distances) if X_train.index[n] != X_instance.index[0]]
        nearest_neighbors = np.array(
            [y_train.iloc[n] for n in nearest_neighbors if X_train.index[n] != X_instance.index[0]])
        assert len(nearest_neighbors) == self.k, (distances, X_instance)
        assert len(distances) == self.k
        return nearest_neighbors, distances

    def compute_weight(self, nearest_neighbors, y_instance):
        y = y_instance.to_numpy()[0]
        return len([n for n in nearest_neighbors if n == y]) / self.k

    def compute_distance_factor(self, X_instance, X_neighbor, distances):
        dist = np.linalg.norm(X_instance.to_numpy().flatten() - X_neighbor.to_numpy().flatten())
        return np.max(distances) / dist

    def compute_random_neighbor(self, X_instance, X_cluster, dataset):
        X_cluster_instance = pd.concat([X_cluster, X_instance], axis=0)
        cat_ord_features = [f for f, t in dataset.feature_types.items() if
                            (t == 'ordinal' or t == 'categorical') and f != dataset.target]
        cat_ord_features = [X_cluster_instance.columns.get_loc(c) for c in cat_ord_features]
        metric = HEOM(X_cluster_instance, cat_ord_features, nan_equivalents=[np.nan])
        # knn = NearestNeighbors(n_neighbors=self.k + 1, metric=metric.heom, n_jobs=-1)
        knn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1)
        knn.fit(X_cluster_instance)

        distances, nearest_neighbors = knn.kneighbors(X_instance)
        distances = distances.flatten()
        nearest_neighbors = nearest_neighbors.flatten()
        nearest_neighbors = np.array([X_cluster_instance.index[n] for n, d in zip(nearest_neighbors, distances) if d > 0])
        assert len(nearest_neighbors) == self.k, (distances, X_instance)
        return nearest_neighbors

