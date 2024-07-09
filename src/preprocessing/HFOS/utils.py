from copy import deepcopy

import numpy as np
import pandas as pd
# from distython import HEOM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset, query_dataset


def get_clusters(dataset: Dataset, data: pd.DataFrame | None = None) -> list:
    """calculates clusters as described in the paper
    :param dataset: dataset
    :return: list of clusters contains:
    group and class,
    the cluster,
    the heterogeneous cluster (same group, different class),
    the heterogeneous cluster (same target, different groups)
    """
    clusters = []
    if data is None:
        data = dataset.train
    for group in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
        query = deepcopy(group)
        query[dataset.target] = dataset.privileged_class
        cluster = query_dataset(query, data)
        group_cluster = compute_heterogeneous_cluster_same_group(dataset, query)
        target_cluster = compute_heterogeneous_cluster_same_target(dataset, query)
        query[dataset.target] = dataset.privileged_class
        clusters.append((deepcopy(query), cluster, group_cluster, target_cluster))

        query[dataset.target] = dataset.unprivileged_class
        cluster = query_dataset(query, data)
        group_cluster = compute_heterogeneous_cluster_same_group(dataset, query)
        target_cluster = compute_heterogeneous_cluster_same_target(dataset, query)
        query[dataset.target] = dataset.unprivileged_class
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
    def __init__(self, k: int, knn, metric, mapping: dict, distance_type='heom'):
        self.k = k
        self.knn = knn
        self.distance_type = distance_type
        self.mapping = mapping
        self.metric = metric

    def generate_example(self, instance: pd.DataFrame, neighbor: pd.DataFrame, dataset: Dataset):
        new_example = {}
        X_instance, y_instance = instance.loc[:, instance.columns != dataset.target], instance[dataset.target]
        X_neighbor, y_neighbor = neighbor.loc[:, neighbor.columns != dataset.target], neighbor[dataset.target]
        group_class_instance = instance[[*dataset.sensitive, dataset.target]].astype(int).astype(str).agg(
            '_'.join, axis=1)
        group_class_neighbor = neighbor[[*dataset.sensitive, dataset.target]].astype(int).astype(str).agg(
            '_'.join, axis=1)
        group_class_instance = pd.DataFrame(np.array([[self.mapping[g] for g in group_class_instance]]).T, columns=['group_class'])
        group_class_neighbor = pd.DataFrame(np.array([[self.mapping[g] for g in group_class_neighbor]]).T, columns=['group_class'])
        if self.distance_type == 'heom':
            knns, dist = self.compute_nearest_neighbors(X_instance, y_instance, dataset)
            distance_factor = self.compute_distance_factor(X_instance, X_neighbor, dist)
        else:
            knns, dist = self.compute_nearest_neighbors(instance, y_instance, dataset)
            distance_factor = self.compute_distance_factor(instance, neighbor, dist)
        weight = self.compute_weight(knns, y_instance)
        weight = dataset.random_state.uniform(low=0, high=weight, size=1)[0]
        for feature in X_instance.columns:
            if dataset.feature_types[feature] in ['continuous', 'ordinal']:
                x1_value = X_instance[feature].to_numpy()[0]
                x2_value = X_neighbor[feature].to_numpy()[0]
                new_value = x1_value + weight * (x1_value - x2_value) * distance_factor
            elif dataset.feature_types[feature] in ['categorical']:
                x1_value = X_instance[feature].to_numpy()[0]
                x2_value = X_neighbor[feature].to_numpy()[0]
                proba = np.array([1, distance_factor])
                new_value = dataset.random_state.choice([x1_value, x2_value], p=proba/np.sum(proba))
            new_example[feature] = [new_value]
        new_example[dataset.target] = [y_instance.to_numpy()[0]]
        return pd.DataFrame(new_example)

    def compute_nearest_neighbors(self, X_instance: pd.DataFrame, y_instance: pd.DataFrame, dataset: Dataset):
        X_train, y_train = dataset.features_and_classes("train")
        if self.distance_type == 'heom':
            distances, nearest_neighbors = self.knn.kneighbors(X_instance)
        else:
            distances, nearest_neighbors = self.knn.kneighbors(X_instance.to_numpy())
        distances = distances.flatten()
        nearest_neighbors = nearest_neighbors.flatten()
        distances = distances[1:]
        nearest_neighbors = nearest_neighbors[1:]
        assert len(nearest_neighbors) == self.k, (distances, X_instance)
        assert len(distances) == self.k
        return nearest_neighbors, distances

    def compute_weight(self, nearest_neighbors, y_instance):
        y = y_instance.to_numpy()[0]
        return len([n for n in nearest_neighbors if n == y]) / self.k

    def compute_distance_factor(self, X_instance, X_neighbor, distances):
        if self.distance_type == 'heom':
            dist = self.metric.heom(X_instance.to_numpy().flatten(), X_neighbor.to_numpy().flatten()).flatten() ** 0.5
        else:
            dist = self.metric.hvdm(X_instance.to_numpy().flatten(), X_neighbor.to_numpy().flatten()).flatten() ** 0.5
        dist = dist if dist != 0 else 1
        return np.max(distances ** 0.5) / dist[0]
