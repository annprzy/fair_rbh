import numpy as np
import pandas as pd
from distython import HEOM, HVDM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset


class FOS_SMOTE:
    def __init__(self, k: int, random_state: int, distance_type='heom'):
        self.k = k
        self.random_state = random_state
        self.distance_type = distance_type

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
                            (t == 'categorical') and f != dataset.target]
        cat_ord_features = [X_base.columns.get_loc(c) for c in cat_ord_features]

        base_instances = neighbors
        X_base_instances = X_neighbors

        group_class = dataset.train[[*dataset.sensitive, dataset.target]].astype(int).astype(str).agg(
            '_'.join, axis=1)
        mapping = {g: i for i, g in enumerate(np.unique(group_class))}
        group_class_m = pd.DataFrame(np.array([[mapping[g] for g in group_class]]).T, columns=['group_class'])
        group_class_neighbors = neighbors[[*dataset.sensitive, dataset.target]].astype(int).astype(str).agg(
            '_'.join, axis=1)
        group_class_neighbors = pd.DataFrame(np.array([[mapping[g] for g in group_class_neighbors]]).T, columns=['group_class'])
        group_class_base = base[[*dataset.sensitive, dataset.target]].astype(int).astype(str).agg(
            '_'.join, axis=1)
        group_class_base = pd.DataFrame(np.array([[mapping[g] for g in group_class_base]]).T,
                                             columns=['group_class'])
        X_train, y_train = dataset.features_and_classes('train')

        if self.k + 1 >= len(X_neighbors):
            k = len(X_neighbors) - 1
        else:
            k = self.k
        if self.distance_type == 'heom':
            metric = HEOM(X_train.to_numpy(), cat_ord_features, nan_equivalents=[np.nan], normalised='')
            knn = NearestNeighbors(n_neighbors=k + 1, metric=metric.heom, n_jobs=-1, algorithm='brute')
            knn.fit(X_neighbors.to_numpy())
        else:
            metric = HVDM(dataset.train.to_numpy(), [X_train.shape[1]], cat_ord_features, nan_equivalents=[np.nan])
            knn = NearestNeighbors(n_neighbors=k + 1, metric=metric.hvdm, n_jobs=-1, algorithm='brute')
            knn.fit(neighbors.to_numpy())
        for idx in range(len(X_base)):
            b = X_base.iloc[[idx], :]
            group_class_b = base[[dataset.target]].iloc[[idx], :]
            if self.distance_type == 'heom':
                distances, nearest_neighbors = knn.kneighbors(b.to_numpy())
            else:
                b_group_class = pd.concat([b.reset_index(drop=True), group_class_b.reset_index(drop=True)], axis=1).to_numpy()
                distances, nearest_neighbors = knn.kneighbors(b_group_class)
            distances = distances.flatten()[1:]
            nearest_neighbors = nearest_neighbors.flatten()[1:]
            nearest_neighbors = X_neighbors.iloc[nearest_neighbors, :]
            assert len(nearest_neighbors) == k, (distances, b, nearest_neighbors, dataset.random_state_init)
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
        random_neighbor = dataset.random_state.choice(list(range(len(nearest_neighbors))), size=1)[0]
        random_neighbor = nearest_neighbors.iloc[[random_neighbor], :]
        for feature in features_names:
            x1_value = base[feature].to_numpy()[0]
            x2_value = random_neighbor[feature].to_numpy()[0]

            if features[feature] in ['continuous', 'ordinal']:
                dif = x1_value - x2_value
                gap = dataset.random_state.random()
                synthetic_example_value = x1_value - gap * dif

            # elif features[feature] == 'ordinal':
            #     synthetic_example_value_float = (x1_value + x2_value) / 2
            #     synthetic_example_value = int(synthetic_example_value_float)

            elif features[feature] == 'categorical':
                # most common value
                all_datapoints = nearest_neighbors
                synthetic_example_value = all_datapoints.mode()[feature][0]
            else:
                exit("Feature type not valid: " + features[feature])

            synthetic_example[feature] = [synthetic_example_value]
        return pd.DataFrame(synthetic_example)
