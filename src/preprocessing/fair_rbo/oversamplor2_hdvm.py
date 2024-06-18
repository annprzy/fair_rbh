import numpy as np
import pandas as pd
from distython import HEOM, HVDM
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters


def run(dataset: Dataset, gamma=0.05):
    rbo = FairRBO(dataset, gamma=gamma, step_size=0.001, n_steps=750, approximate_potential=True,
                  n_nearest_neighbors=35, k=5)
    new_data = rbo.fit_sample()
    dataset.set_fair(new_data)


def distance(x, y, metric):
    return metric.hvdm(x, y).flatten()[0]
    #return metric.heom(x, y).flatten()[0]


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, neighbors, groups_neighbors, class_neighbors, group_current, class_current, metric, gamma, mapping,
                           return_distances=False):
    result = 0.0
    distances = []
    group_class_point = '_'.join([group_current, str(int(class_current))])
    group = group_current.split('_')
    for i, n_point in enumerate(neighbors):
        group_class_n = '_'.join([groups_neighbors[i], str(int(class_neighbors[i]))])
        dist = distance(np.append(point, mapping[group_class_point]), np.append(n_point, mapping[group_class_n]), metric)

        group_neighbor = groups_neighbors[i].split('_')
        same_groups = np.array([1 if i == j else 0 for i, j in zip(group, group_neighbor)])
        multiplier = 0.5 + (np.sum(same_groups) / len(same_groups)) * 0.5
        if class_neighbors[i] == class_current:
            rbf_score = rbf(dist, gamma)
            result -= rbf_score
        else:
            rbf_score = rbf(dist, gamma)
            result += rbf_score

    if return_distances:
        return result, distances
    return result


def generate_possible_directions(point, included_features, cat_features, cat_freq, dataset, excluded_direction=None):
    possible_directions = []

    for dimension in included_features:
        if dimension in cat_features:
            possible_values = cat_freq[dimension][point[dimension]]
            unique_values = np.unique(possible_values)
            for v in unique_values:
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
                 n_nearest_neighbors=25, k=3):
        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.k = k
        self.dataset = dataset

    def calculate_frequencies_neighbors(self, X: np.ndarray, features: list, metric):
        result = {f: {} for f in features}
        knn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1, metric=metric.hvdm, algorithm='brute').fit(X)
        for x in X:
            nns = knn.kneighbors([x], return_distance=False)
            nns = nns.flatten()
            nns = X[nns, :]
            for f in features:
                if x[f] not in result[f]:
                    result[f][x[f]] = [n[f] for n in nns]
                else:
                    result[f][x[f]].extend([n[f] for n in nns])
        return result

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
        metric = HVDM(pd.concat([X_train, group_class_m], axis=1).to_numpy(), [X_train.shape[1]], cat_ord_features,
                      nan_equivalents=[np.nan], normalised='')
        #metric = HEOM(X_train.to_numpy(), cat_ord_features, nan_equivalents=[np.nan], normalised='')

        groups = np.array(X_train[self.dataset.sensitive].astype(int).astype(str).agg('_'.join, axis=1).to_list())

        clusters = get_clusters(self.dataset, self.dataset.train)
        len_clusters = [len(entry[1]) for entry in clusters]
        #max_len = {self.dataset.majority: np.max([len(entry[1]) for entry in clusters if entry[0][self.dataset.target] == self.dataset.majority]), self.dataset.minority: np.max([len(entry[1]) for entry in clusters if entry[0][self.dataset.target] == self.dataset.minority])}

        max_len = np.max(len_clusters)

        cat_frequencies = self.calculate_frequencies_neighbors(pd.concat([X_train, group_class_m], axis=1).to_numpy(), list(np.argwhere(types_vector == 0).flatten()), metric)
        appended = []

        for e, entry in enumerate(clusters):
            new_examples = []
            query, cluster_pd, _, _ = entry
            current_group = '_'.join([str(int(query[s])) for s in self.dataset.sensitive])
            X = cluster_pd.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
            y = cluster_pd[self.dataset.target].to_numpy()
            cluster = cluster_pd.to_numpy()

            current_class = query[self.dataset.target]
            group_class_current_str = '_'.join([current_group, str(int(current_class))])
            group_class_current = mapping[group_class_current_str]

            n = max_len - len(cluster)

            considered_points_indices = range(len(y))

            n_synthetic_points_per_object = {i: 0 for i in considered_points_indices}

            if n > 0:
                for _ in range(n):
                    idx = self.dataset.random_state.choice(considered_points_indices)
                    n_synthetic_points_per_object[idx] += 1

                for i in considered_points_indices:
                    if n_synthetic_points_per_object[i] == 0:
                        continue

                    point = X[i]

                    distance_vector = [distance(np.append(point, group_class_current), np.append(x, group_class_m.to_numpy()[xx]), metric) for xx, x in enumerate(X_full)]
                    distance_vector[i] = -np.inf
                    indices = np.argsort(distance_vector)[1:(self.n_nearest_neighbors + 2)]

                    closest_points = X_full[indices]
                    closest_groups = groups[indices]
                    closest_y = y_full[indices]

                    for _ in range(n_synthetic_points_per_object[i]):
                        translation = point.copy()
                        translation_history = [translation]
                        potential = mutual_class_potential(point, closest_points, closest_groups, closest_y, current_group, current_class, metric, self.gamma, mapping)
                        possible_directions = generate_possible_directions(point, sampling_features,
                                                                           list(np.argwhere(types_vector == 0).flatten()),
                                                                           cat_frequencies, self.dataset)

                        for _ in range(self.n_steps):
                            if len(possible_directions) == 0:
                                break

                            dimension, sign = possible_directions.pop()
                            modified_point = point.copy()
                            if types_vector[dimension] == 0:
                                modified_point[dimension] = sign
                            else:
                                full_range = upper_range[dimension] - lower_range[dimension]
                                modified_point[dimension] += sign * full_range * self.step_size

                            modified_potential = mutual_class_potential(modified_point, closest_points, closest_groups, closest_y, current_group, current_class, metric, self.gamma, mapping)
                            s_id = [ss for ss, s in enumerate(X_train.columns) if s in self.dataset.sensitive][0]
                            assert dimension != s_id
                            assert point[s_id] == modified_point[s_id]

                            if np.abs(modified_potential) < np.abs(potential):
                                translation = modified_point
                                translation_history.append(translation)
                                potential = modified_potential
                                if types_vector[dimension] == 0:
                                    excluded = (dimension, sign)
                                else:
                                    excluded = (dimension, -sign)

                                possible_directions = generate_possible_directions(modified_point, sampling_features,
                                                                                   list(np.argwhere(types_vector == 0).flatten()),
                                                                                   cat_frequencies, self.dataset,
                                                                                   excluded_direction=excluded)

                        new_examples.append(translation)
                new_examples = np.array(new_examples)
                new_X = np.concatenate((X, new_examples))
                new_y = np.concatenate([y, current_class * np.ones(len(new_examples))])
                new_data = np.c_[new_X, new_y]
            else:
                new_data = np.c_[X, y]
            appended.append(new_data)

        new_data_full = np.concatenate(appended)
        new_data_full = pd.DataFrame(new_data_full, columns=self.dataset.train.columns)

        return new_data_full
