import networkx as nx
import numpy as np
import pandas as pd
from distython import HEOM, HVDM
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters


def run(dataset: Dataset, gamma=0.05, approach_number=0, distance_type='heom'):
    rbo = FairRBO(dataset, gamma=gamma, step_size=0.002, n_steps=30, approximate_potential=True,
                  n_nearest_neighbors=50, k=5, approach_number=approach_number, distance_type=distance_type)
    new_data = rbo.fit_sample()
    dataset.set_fair(new_data)


def distance(x, y, metric, x_class=None, y_class=None, distance_type='heom'):
    if distance_type == 'heom':
        return metric.heom(x, y).flatten()[0] ** 0.5
    else:
        return metric.hvdm(np.append(x, x_class), np.append(y, y_class)).flatten()[0] ** 0.5


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, neighbors, groups_neighbors, class_neighbors, group_current, class_current, metric,
                           gamma, return_distances=False, distance_type='heom', mapping=None):
    result = 0.0
    distances = []
    group = group_current.split('_')
    for i, n_point in enumerate(neighbors):
        #print(point, n_point, dist, rbf(dist, gamma))
        group_neighbor = groups_neighbors[i].split('_')
        same_groups = np.array([1 if i == j else 0 for i, j in zip(group, group_neighbor)])

        group_class_current = '_'.join([group_current, str(int(class_current))])
        group_class_neighbor = '_'.join([groups_neighbors[i], str(int(class_neighbors[i]))])

        dist = distance(point, n_point, metric, x_class=class_current, y_class=class_neighbors[i], distance_type=distance_type)
        # multiplier = (np.sum(same_groups) / len(same_groups))
        rbf_score = rbf(dist, gamma)
        # if class_neighbors[i] == class_current and np.sum(same_groups) == len(same_groups):
        #     result -= rbf_score
        if group_class_current == group_class_neighbor:
            result -= rbf_score
        else:
            result += rbf_score

    if return_distances:
        return result, distances
    return result


def generate_possible_directions(point, included_features, cat_features, k_nearest_neighbors, dataset, excluded_direction=None):
    possible_directions = []

    for dimension in included_features:
        if dimension in cat_features:
            possible_values = k_nearest_neighbors[:, dimension].flatten()
            unique_values, counts = np.unique(possible_values, return_counts=True)
            counts = np.argsort(-counts)
            unique_values = unique_values[counts]
            for v in unique_values[:2]:
                if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction[1] != v):
                    possible_directions.append((dimension, v))
        else:
            for sign in [-1, 1]:
                if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction[1] != sign):
                    possible_directions.append((dimension, sign))

    dataset.random_state.shuffle(possible_directions)

    return possible_directions


class FairRBO:
    def __init__(self, dataset: Dataset, gamma=0.05, step_size=0.001, n_steps=500, approximate_potential=True,
                 n_nearest_neighbors=25, k=5, approach_number=0, distance_type='heom'):
        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.k = k
        self.dataset = dataset
        self.approach_number = approach_number
        self.distance_type = distance_type

    def get_knn_struct(self, X: np.ndarray, metric, mapping_groups=None):
        if self.distance_type == 'heom':
            knn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1, metric=metric.heom, algorithm='brute').fit(X)
        else:
            knn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1, metric=metric.hvdm, algorithm='brute').fit(np.c_[X, mapping_groups])
        return knn

    def get_knns(self, knn, x, mapping_group=None):
        if self.distance_type == 'heom':
            nns = knn.kneighbors([x], return_distance=False)
        else:
            nns = knn.kneighbors([np.append(x, mapping_group)], return_distance=False)
        nns = nns.flatten()[1:]
        return nns


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
        len_clusters = [len(entry[1]) for entry in clusters]

        order_clusters = np.argsort(len_clusters)[::-1]
        max_len = np.max(len_clusters)

        group_class_m_np = np.array([mapping[g] for g in group_class])

        knn_cat = self.get_knn_struct(X_full, metric, group_class_m_np)

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

            n = max_len - len(cluster)

            if self.approach_number == 0: # representative sample
                samples = []
                groups_sample = []
                for i in range(0, e):
                    cluster_e = clusters[order_clusters[i]][1].to_numpy()
                    query_e = clusters[order_clusters[i]][0]
                    if i - 1 >= 0:
                        appended_e = appended[i - 1]
                        cluster_e = appended_e
                    sample = self.dataset.random_state.choice(range(len(cluster_e)), replace=False, size=max_len//e).flatten()
                    sample = cluster_e[sample]
                    samples.append(sample)
                    groups_sample.extend(['_'.join([str(int(query_e[s])) for s in self.dataset.sensitive])] * len(sample))
                samples.append(cluster)
                groups_sample.extend([current_group] * len(cluster))
                samples = np.concatenate(samples)
                groups_sample = np.array(groups_sample)
                X_sample = samples[:, :-1]
                y_sample = samples[:, -1]
            elif self.approach_number == 1: #one-vs-all
                if len(appended) > 0:
                    new_data_now = np.concatenate(appended)
                    new_data_now = pd.DataFrame(new_data_now, columns=self.dataset.train.columns)
                    not_oversampled_clusters = [clusters[order_clusters[i]][1] for i in range(e, len(clusters))]
                    whole_data_now = pd.concat([*not_oversampled_clusters, new_data_now])
                    X_whole_now = whole_data_now.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
                    y_whole_now = whole_data_now[self.dataset.target].to_numpy()
                    groups_now = np.array(whole_data_now[self.dataset.sensitive].astype(int).astype(str).agg('_'.join, axis=1).to_list())
                    X_sample = X_whole_now
                    y_sample = y_whole_now
                    groups_sample = groups_now
                else:
                    X_sample = X_full
                    y_sample = y_full
                    groups_sample = groups
            elif self.approach_number == 2: #k nearest neighbors from a given group
                if len(appended) > 0:
                    new_data_now = np.concatenate(appended)
                    new_data_now = pd.DataFrame(new_data_now, columns=self.dataset.train.columns)
                    not_oversampled_clusters = [clusters[order_clusters[i]][1] for i in range(e, len(clusters))]
                    whole_data_now = pd.concat([*not_oversampled_clusters, new_data_now])
                    X_whole_now = whole_data_now.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
                    y_whole_now = whole_data_now[self.dataset.target].to_numpy()
                    groups_now = np.array(
                        whole_data_now[self.dataset.sensitive].astype(int).astype(str).agg('_'.join, axis=1).to_list())
                    X_sample = X_whole_now
                    y_sample = y_whole_now
                    groups_sample = groups_now
                else:
                    X_sample = X_full
                    y_sample = y_full
                    groups_sample = groups
            elif self.approach_number == 3: #only subgroups from a given class
                if len(appended) > 0:
                    new_data_now = np.concatenate(appended)
                    new_data_now = pd.DataFrame(new_data_now, columns=self.dataset.train.columns)
                    not_oversampled_clusters = [clusters[order_clusters[i]][1] for i in range(e, len(clusters))]
                    whole_data_now = pd.concat([*not_oversampled_clusters, new_data_now])
                    X_whole_now = whole_data_now.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
                    y_whole_now = whole_data_now[self.dataset.target].to_numpy()
                    groups_now = np.array(
                        whole_data_now[self.dataset.sensitive].astype(int).astype(str).agg('_'.join, axis=1).to_list())
                    X_sample = X_whole_now[y_whole_now == current_class]
                    y_sample = y_whole_now[y_whole_now == current_class]
                    groups_sample = groups_now[y_whole_now == current_class]
                else:
                    X_sample = X_full[y_full == current_class]
                    y_sample = y_full[y_full == current_class]
                    groups_sample = groups[y_full == current_class]
            else: #only subgroups from a given group
                if len(appended) > 0:
                    new_data_now = np.concatenate(appended)
                    new_data_now = pd.DataFrame(new_data_now, columns=self.dataset.train.columns)
                    not_oversampled_clusters = [clusters[order_clusters[i]][1] for i in range(e, len(clusters))]
                    whole_data_now = pd.concat([*not_oversampled_clusters, new_data_now])
                    X_whole_now = whole_data_now.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
                    y_whole_now = whole_data_now[self.dataset.target].to_numpy()
                    groups_now = np.array(
                        whole_data_now[self.dataset.sensitive].astype(int).astype(str).agg('_'.join, axis=1).to_list())
                    X_sample = X_whole_now[groups_now == current_group]
                    y_sample = y_whole_now[groups_now == current_group]
                    groups_sample = groups_now[groups_now == current_group]
                else:
                    X_sample = X_full[groups == current_group]
                    y_sample = y_full[groups == current_group]
                    groups_sample = groups[groups == current_group]

            considered_points_indices = range(len(y))

            group_class_sample = ['_'.join([groups_sample[i], str(int(y_sample[i]))]) for i in range(len(groups_sample))]
            # print(group_class_sample)

            n_synthetic_points_per_object = {i: 0 for i in considered_points_indices}

            if n > 0:

                for _ in range(n):
                    idx = self.dataset.random_state.choice(considered_points_indices)
                    n_synthetic_points_per_object[idx] += 1

                for i in considered_points_indices:
                    if n_synthetic_points_per_object[i] == 0:
                        continue

                    point = X[i]

                    if self.approach_number in [0, 1, 3, 4]:

                        if self.distance_type == 'heom':
                            used_metric = metric
                            distance_vector = [metric.heom(point, x) for x in X_sample]
                        else:
                            used_metric = metric
                            distance_vector = [metric.hvdm(np.append(point, current_class), np.append(x, y_x)) for x, y_x in zip(X_sample, y_sample)]
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
                                                               np.append(x, query_j[self.dataset.target])) for x in X_cluster_j]

                            indices = np.argsort(distance_vector)[1:(num_neighbors_per_subgroup + 1)]
                            closest_points.append(X_cluster_j[indices])
                            closest_groups.extend([group_j] * len(indices))
                            closest_y.extend(y_cluster_j[indices])

                        closest_points = np.concatenate(closest_points)
                        closest_groups = np.array(closest_groups)
                        closest_y = np.array(closest_y)

                    for _ in range(n_synthetic_points_per_object[i]):
                        translation = point.copy()
                        k_closest_neighbors = self.get_knns(knn_cat, translation, mapping_group=mapping[group_class_current])
                        k_closest_neighbors = X_full[k_closest_neighbors, :]
                        translation_history = [translation]
                        potential = mutual_class_potential(point, closest_points, closest_groups, closest_y,
                                                           current_group, current_class, used_metric, self.gamma, distance_type=self.distance_type, mapping=mapping)
                        possible_directions = generate_possible_directions(point, sampling_features,
                                                                           list(np.argwhere(
                                                                               types_vector == 0).flatten()),
                                                                           k_closest_neighbors, self.dataset)

                        for _ in range(self.n_steps):
                            modified_point = translation.copy()
                            if len(possible_directions) == 0:
                                break
                            dimension, sign = possible_directions.pop()

                            if types_vector[dimension] == 0:
                                modified_point[dimension] = sign
                            else:
                                full_range = upper_range[dimension] - lower_range[dimension]
                                v = modified_point[dimension] + sign * full_range * self.step_size
                                if v > upper_range[dimension]:
                                    v = upper_range[dimension] + (v - upper_range[dimension])
                                elif v < lower_range[dimension]:
                                    v = lower_range[dimension] + (lower_range[dimension] - v)
                                modified_point[dimension] = v

                            modified_potential = mutual_class_potential(modified_point, closest_points, closest_groups,
                                                                        closest_y, current_group, current_class,
                                                                        used_metric, self.gamma, distance_type=self.distance_type, mapping=mapping)
                            s_id = [ss for ss, s in enumerate(X_train.columns) if s in self.dataset.sensitive][0]
                            assert dimension != s_id
                            assert point[s_id] == modified_point[s_id]

                            if np.abs(modified_potential) < np.abs(potential):
                                translation = modified_point.copy()
                                translation_history.append(translation)
                                potential = modified_potential
                                if types_vector[dimension] == 0:
                                    excluded = (dimension, sign)
                                else:
                                    excluded = (dimension, -sign)
                                k_closest_neighbors = self.get_knns(knn_cat, translation,
                                                                    mapping_group=current_class)
                                k_closest_neighbors = X_full[k_closest_neighbors, :]
                                possible_directions = generate_possible_directions(modified_point, sampling_features,
                                                                                   list(np.argwhere(
                                                                                       types_vector == 0).flatten()),
                                                                                   k_closest_neighbors, self.dataset,
                                                                                   excluded_direction=excluded)

                        new_examples.append(translation.copy())
                new_examples = np.array(new_examples)
                new_X = np.concatenate((X, new_examples))
                new_y = np.concatenate([y, current_class * np.ones(len(new_examples))])
                new_data = np.c_[new_X, new_y]
            else:
                new_data = np.c_[X, y]
            appended.append(new_data)

        new_data_full = np.concatenate(appended)
        new_data_full = pd.DataFrame(new_data_full, columns=[*[f for f in self.dataset.train.columns if f != self.dataset.target], self.dataset.target])

        return new_data_full
