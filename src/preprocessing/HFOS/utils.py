from copy import deepcopy

import numpy as np
import pandas as pd
# from distython import HEOM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset, query_dataset


def get_clusters(dataset: Dataset) -> list:
    """calculates clusters as described in the paper
    :param dataset: dataset
    :return: list of clusters contains:
    group and class,
    the cluster,
    the heterogeneous cluster (same group, different class),
    the heterogeneous cluster (same target, different groups)
    """
    clusters = []
    for group in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
        query = deepcopy(group)
        query[dataset.target] = dataset.privileged_class
        cluster = query_dataset(query, dataset.train)
        group_cluster = compute_heterogeneous_cluster_same_group(dataset, query)
        target_cluster = compute_heterogeneous_cluster_same_target(dataset, query)
        clusters.append((query, cluster, group_cluster, target_cluster))

        query[dataset.target] = dataset.unprivileged_class
        cluster = query_dataset(query, dataset.train)
        group_cluster = compute_heterogeneous_cluster_same_group(dataset, query)
        target_cluster = compute_heterogeneous_cluster_same_target(dataset, query)
        clusters.append((query, cluster, group_cluster, target_cluster))

    return clusters


def compute_heterogeneous_cluster_same_group(dataset: Dataset, query: dict) -> pd.DataFrame:
    query[dataset.target] = dataset.privileged_class if query[
                                                            dataset.target] == dataset.unprivileged_class else dataset.unprivileged_class
    df = query_dataset(query, dataset.train)
    return df


def compute_heterogeneous_cluster_same_target(dataset: Dataset, query: dict) -> pd.DataFrame:
    query_str = [f'`{key}`!={value}' if type(value) is not str else f'`{key}`!="{value}"' for key, value in
                 query.items() if key != dataset.target]
    query_str = ' or '.join(query_str)
    target_query_str = f'`{dataset.target}`=={query[dataset.target]}' if type(
        query[dataset.target]) is not str else f'`{dataset.target}`=="{query[dataset.target]}"'
    target_query_str = '(' + target_query_str + ')'
    query_str = '(' + query_str + ')' + ' and ' + target_query_str
    df = dataset.train.query(query_str)
    return df


class HFOS_SMOTE:
    def __init__(self, k: int, knn):
        self.k = k
        self.knn = knn

    def generate_example(self, instance: pd.DataFrame, neighbor: pd.DataFrame, dataset: Dataset):
        new_example = {}
        X_instance, y_instance = instance.loc[:, instance.columns != dataset.target], instance[dataset.target]
        X_neighbor, y_neighbor = neighbor.loc[:, neighbor.columns != dataset.target], neighbor[dataset.target]

        knns, dist = self.compute_nearest_neighbors(X_instance, y_instance, dataset)
        weight = self.compute_weight(knns, y_instance)
        weight = dataset.random_state.uniform(low=0, high=weight, size=1)[0]
        distance_factor = self.compute_distance_factor(X_instance, X_neighbor, dist)
        for feature in X_instance.columns:
            # for now only continuous data are handled (from equation described in the paper)
            x1_value = X_instance[feature].to_numpy()[0]
            x2_value = X_neighbor[feature].to_numpy()[0]
            new_value = x1_value + weight * (x2_value - x1_value) * distance_factor
            # if dataset.feature_types[feature] in ['categorical', 'ordinal']:
            #     new_value = int(new_value)
            new_example[feature] = [new_value]
        new_example[dataset.target] = [y_instance.to_numpy()[0]]
        return pd.DataFrame(new_example)

    def compute_nearest_neighbors(self, X_instance: pd.DataFrame, y_instance: pd.DataFrame, dataset: Dataset):
        X_train, y_train = dataset.features_and_classes("train")
        #
        # knn = NearestNeighbors(n_neighbors=self.k + 1, p=2)
        # knn.fit(X_train)
        # cat_ord_features = [f for f, t in dataset.feature_types.items() if
        #                     (t == 'ordinal' or t == 'categorical') and f != dataset.target]
        # cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
        # metric = HEOM(X_train, cat_ord_features, nan_equivalents=[np.nan])
        # knn = NearestNeighbors(n_neighbors=self.k + 1, metric=metric.heom)
        # knn.fit(X_train)

        distances, nearest_neighbors = self.knn.kneighbors(X_instance)
        distances = distances.flatten()
        nearest_neighbors = nearest_neighbors.flatten()
        distances = [d for n, d in zip(nearest_neighbors, distances) if X_train.index[n] != X_instance.index[0]]
        nearest_neighbors = np.array([y_train.iloc[n] for n in nearest_neighbors if X_train.index[n] != X_instance.index[0]])
        assert len(nearest_neighbors) == self.k, (distances, X_instance)
        assert len(distances) == self.k
        return nearest_neighbors, distances

    def compute_weight(self, nearest_neighbors, y_instance):
        y = y_instance.to_numpy()[0]
        return len([n for n in nearest_neighbors if n == y]) / self.k

    def compute_distance_factor(self, X_instance, X_neighbor, distances):
        dist = np.linalg.norm(X_instance.to_numpy().flatten() - X_neighbor.to_numpy().flatten())
        return np.max(distances) / dist
