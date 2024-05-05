import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset


def compute_classes_groups(dataset: Dataset) -> list:
    groups_maj_min = []
    for group in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
        query = [f'`{key}`=={value}' if type(value) is not str else f'`{key}`=="{value}"' for key, value in
                 group.items()]
        query = ' and '.join(query)
        group_subset = dataset.train.query(query)
        majority, minority = dataset.compute_min_maj_class(group_subset)
        groups_maj_min.append({**group, 'majority': majority, 'minority': minority})
    return groups_maj_min


class SPIDER:
    def __init__(self, knn, k, relabel):
        self.knn = knn
        self.k = k
        self.relabel = relabel

    def classify(self, data: pd.DataFrame, dataset: Dataset) -> tuple[np.array, np.array]:
        not_safe = []
        neighbors = []
        X_train, y_train = dataset.features_and_classes('train')
        X_data = data.loc[:, ~data.columns.isin([dataset.target, *dataset.sensitive])]
        nearest_neighbors = self.knn.kneighbors(X_data, return_distance=False)
        for neigh, instance_id in zip(nearest_neighbors, data.index):
            neigh = np.array(
                [y_train.iloc[n] for n in neigh if X_train.index[n] != instance_id])
            class_nns = y_train[neigh]
            group_nns = X_train.loc[neigh, dataset.sensitive]
            neighbors.append(neigh)
            class_instance = y_train[instance_id]
            group_instance = X_train.loc[instance_id, dataset.sensitive].values
            i = 0
            for c, g in zip(class_nns.values, group_nns.values):
                if c != class_instance:
                    i += 1
                else:
                    for g_n, g_i in zip(g,group_instance):
                        if g_n != g_i:
                            i += 1
                            break

            if i > self.k / 2:
                not_safe.append(instance_id)
                neighbors.append(neigh)
        return np.array(not_safe), np.array(neighbors)

    def undersampling(self, data: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
        not_safe, _ = self.classify(data, dataset)
        new_data = data.drop(not_safe)
        return new_data

    def oversampling(self, data: pd.DataFrame, dataset: Dataset, group: dict, num_examples_per_instance: int) -> pd.DataFrame:
        not_safe, neighbors = self.classify(data, dataset)
        new_data = []
        for instance_id, nns in zip(not_safe, neighbors):
            for _ in range(num_examples_per_instance):
                example = self.generate_example(instance_id, nns, dataset, group)
                new_data.append(example)
        return pd.DataFrame(pd.concat([*new_data, data]))

    def generate_example(self, instance: int, nearest_neighbors: np.array, dataset: Dataset, group: dict) -> pd.DataFrame:
        synthetic_example = {}
        X_train, y_train = dataset.features_and_classes('train')
        nearest = X_train.loc[nearest_neighbors, :]
        random_neighbor = dataset.random_state.choice(nearest_neighbors, size=1)[0]
        random_neighbor = X_train.loc[[random_neighbor], :]
        base = X_train.loc[[instance], :]
        gap = dataset.random_state.random()

        features_names = base.columns
        features = dataset.feature_types

        for feature in features_names:
            x1_value = base[feature].to_numpy()[0]
            x2_value = random_neighbor[feature].to_numpy()[0]

            if features[feature] == 'continuous':
                dif = x1_value - x2_value
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
        for feature in group:
            synthetic_example[feature] = group[feature]
        return pd.DataFrame(synthetic_example)
