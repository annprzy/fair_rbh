from copy import deepcopy

import numpy as np
import pandas as pd
from distython import HVDM, HEOM

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters
from src.preprocessing.fair_rbo.oversamplor2 import mutual_class_potential, FairRBO, distance, rbf


def run(dataset: Dataset, gamma=0.05, approach_number=0, distance_type='heom'):
    rbu = FairRBU(dataset, gamma=gamma, n_nearest_neighbors=50, approach_number=approach_number,
                  distance_type=distance_type)
    new_data = rbu.fit_sample()
    dataset.train = new_data
    rbo = FairRBO(dataset, gamma=gamma, step_size=0.001, n_steps=100, approximate_potential=True,
                  n_nearest_neighbors=50, k=5, approach_number=approach_number, distance_type=distance_type)
    new_data = rbo.fit_sample()
    dataset.set_fair(new_data)


def run_under(dataset: Dataset, gamma=0.05, approach_number=0, distance_type='heom'):
    rbu = FairRBU(dataset, gamma=gamma, n_nearest_neighbors=50, approach_number=approach_number,
                  distance_type=distance_type, type_alg='under')
    new_data = rbu.fit_sample()
    dataset.set_fair(new_data)


class FairRBU:
    def __init__(self, dataset: Dataset, gamma=0.05, approximate_potential=True, type_alg='hybrid',
                 n_nearest_neighbors=25, approach_number=0, distance_type='heom'):
        self.gamma = gamma
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.dataset = dataset
        self.approach_number = approach_number
        self.distance_type = distance_type
        self.type_alg = type_alg

    def fit_sample(self):
        cat_ord_features = [f for f, t in self.dataset.feature_types.items() if
                            (t == 'categorical') and f not in [self.dataset.target]]
        X_train, y_train = self.dataset.features_and_classes('train')
        X_full, y_full = X_train.to_numpy(), y_train.to_numpy()
        types_vector = np.array([0 if self.dataset.feature_types[c] in ['categorical'] else 1 for c in X_train.columns])
        lower_range = X_train.min(axis='columns').to_numpy().flatten().T
        upper_range = X_train.max(axis='columns').to_numpy().flatten().T
        sampling_features = [i for i, c in enumerate(X_train.columns) if c not in self.dataset.sensitive]

        cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
        group_class = self.dataset.train[[*self.dataset.sensitive, self.dataset.target]].astype(int).astype(str).agg(
            '_'.join, axis=1)
        unique_groups = np.unique(group_class)

        mapping = {g: i for i, g in enumerate(unique_groups)}
        group_class_m = pd.DataFrame(np.array([[mapping[g] for g in group_class]]).T, columns=['group_class'])

        if self.distance_type == 'heom':
            metric = HEOM(X_train.to_numpy(), cat_ord_features, nan_equivalents=[np.nan], normalised='')
        else:
            metric = HVDM(self.dataset.train.to_numpy(), [X_train.shape[1]], cat_ord_features,
                          nan_equivalents=[np.nan])

        groups = np.array(X_train[self.dataset.sensitive].astype(int).astype(str).agg('_'.join, axis=1).to_list())

        clusters = get_clusters(self.dataset, self.dataset.train)
        len_clusters = np.array([len(entry[1]) for entry in clusters])

        order_clusters = np.argsort(len_clusters)
        len_to_achieve = int(np.median(len_clusters))
        #len_to_achieve = int(np.mean(len_clusters))

        group_class_m_np = np.array([mapping[g] for g in group_class])

        appended = []

        for e, ee in enumerate(order_clusters):
            entry = clusters[ee]
            new_examples = []
            query, cluster_pd, _, _ = entry
            current_group = '_'.join([str(int(query[s])) for s in self.dataset.sensitive])
            X = cluster_pd.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
            y = cluster_pd[self.dataset.target].to_numpy()
            cluster = cluster_pd.to_numpy()

            current_class = query[self.dataset.target]
            group_class_current = '_'.join([current_group, str(int(current_class))])
            # group_class_current = mapping[group_class_current_str]

            n = len(cluster) - len_to_achieve

            if self.approach_number == 0:  # representative sample
                samples = []
                groups_sample = []
                for i in range(0, e):
                    cluster_e = clusters[order_clusters[i]][1].to_numpy()
                    query_e = clusters[order_clusters[i]][0]
                    if len(cluster_e) > len_to_achieve // e:
                        sample = self.dataset.random_state.choice(range(len(cluster_e)), replace=False,
                                                                  size=len_to_achieve // e).flatten()
                        sample = cluster_e[sample]
                    else:
                        sample = deepcopy(cluster_e)
                    samples.append(sample)
                    groups_sample.extend(
                        ['_'.join([str(int(query_e[s])) for s in self.dataset.sensitive])] * len(sample))
                samples.append(cluster)
                groups_sample.extend([current_group] * len(cluster))
                samples = np.concatenate(samples)
                groups_sample = np.array(groups_sample)
                X_sample = samples[:, :-1]
                y_sample = samples[:, -1]
            elif self.approach_number == 1:  #one-vs-all
                X_sample = X_full
                y_sample = y_full
                groups_sample = groups
            elif self.approach_number == 2:  #k nearest neighbors from a given group
                X_sample = X_full
                y_sample = y_full
                groups_sample = groups
            elif self.approach_number == 3:  #only subgroups from a given class
                X_sample = X_full[y_full == current_class]
                y_sample = y_full[y_full == current_class]
                groups_sample = groups[y_full == current_class]
            else:  #only subgroups from a given group
                X_sample = X_full[groups == current_group]
                y_sample = y_full[groups == current_group]
                groups_sample = groups[groups == current_group]

            considered_points_indices = range(len(y))

            group_class_sample = ['_'.join([groups_sample[i], str(int(y_sample[i]))]) for i in
                                  range(len(groups_sample))]

            if n > 0:
                potentials = []
                for i in considered_points_indices:
                    point = X[i]

                    if self.approach_number in [0, 1, 3, 4]:

                        if self.distance_type == 'heom':
                            used_metric = metric
                            distance_vector = [metric.heom(point, x) for x in X_sample]
                        else:
                            used_metric = metric
                            distance_vector = [metric.hvdm(np.append(point, current_class), np.append(x, y_x)) for
                                               x, y_x in zip(X_sample, y_sample)]
                        indices = np.argsort(distance_vector)[1:(self.n_nearest_neighbors + 1)]
                        # print(self.distance_type, distance_vector)
                        closest_points = X_sample[indices]
                        closest_groups = groups_sample[indices]
                        closest_y = y_sample[indices]

                    else:
                        num_neighbors_per_subgroup = self.n_nearest_neighbors // len(order_clusters)
                        closest_points = []
                        closest_groups = []
                        closest_y = []
                        for j in clusters:
                            query_j, cluster_j, _, _ = j
                            group_j = '_'.join([str(int(query_j[s])) for s in self.dataset.sensitive])
                            X_cluster_j = cluster_pd.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
                            y_cluster_j = cluster_pd[self.dataset.target].to_numpy()
                            # distance_vector = [metric.heom(point, x) for x in X_cluster_j]
                            group_class_cluster_j = '_'.join([group_j, str(int(query[self.dataset.target]))])

                            if self.distance_type == 'heom':
                                used_metric = metric
                                distance_vector = [metric.heom(point, x) for x in X_cluster_j]
                            else:
                                used_metric = metric
                                distance_vector = [metric.hvdm(np.append(point, current_class),
                                                               np.append(x, query_j[self.dataset.target])) for x in
                                                   X_cluster_j]

                            indices = np.argsort(distance_vector)[1:(num_neighbors_per_subgroup + 1)]
                            closest_points.append(X_cluster_j[indices])
                            closest_groups.extend([group_j] * len(indices))
                            closest_y.extend(y_cluster_j[indices])

                        closest_points = np.concatenate(closest_points)
                        closest_groups = np.array(closest_groups)
                        closest_y = np.array(closest_y)

                    potential = mutual_class_potential(point, closest_points, closest_groups, closest_y,
                                                       current_group, current_class, used_metric, self.gamma,
                                                       distance_type=self.distance_type, mapping=mapping)

                    potentials.append(potential)
                potentials = np.array(potentials)
                sorted_potentials = np.argsort(-potentials).flatten()
                to_remain = sorted_potentials[n:]
                new_X = X[to_remain, :]
                new_y = y[to_remain]
                new_data = np.c_[new_X, new_y]
            else:
                new_data = np.c_[X, y]
            appended.append(new_data)

        new_data_full = np.concatenate(appended)
        new_data_full = pd.DataFrame(new_data_full, columns=[*[f for f in self.dataset.train.columns if f != self.dataset.target], self.dataset.target])

        return new_data_full
