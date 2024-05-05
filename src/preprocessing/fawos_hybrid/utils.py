import numpy as np
import pandas as pd
from distython import HEOM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset
from src.preprocessing.FOS.utils import FOS_SMOTE


def analyze_neighborhood(dataset: Dataset, k: int = 5) -> tuple:
    group_and_class = dataset.train[[*dataset.sensitive, dataset.target]].astype(str).agg('-'.join, axis=1)
    X_train_no_group = dataset.train.loc[:, ~dataset.train.columns.isin([dataset.target, *dataset.sensitive])]
    cat_ord_features = [f for f in X_train_no_group.columns if
                        (dataset.feature_types[f] == 'ordinal' or dataset.feature_types[f] == 'categorical')]
    cat_ord_features = [X_train_no_group.columns.get_loc(c) for c in cat_ord_features]
    metric = HEOM(X_train_no_group, cat_ord_features, nan_equivalents=[np.nan])
    knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1, metric=metric.heom).fit(X_train_no_group)

    neighbors = knn.kneighbors(X_train_no_group, return_distance=False)
    neighbors = neighbors[:, 1:]
    safe_neighbors = []
    categories = []
    for i, neighbor in enumerate(neighbors):
        current_cat = group_and_class.iloc[i]
        safe_neighbor = [n for n in neighbor if group_and_class.iloc[n] == current_cat]
        safe_neighbors.append(safe_neighbor)

    for i, neighbor in enumerate(safe_neighbors):
        if len(neighbor) == 0:
            categories.append('outlier')
        elif len(neighbor) == 1:
            if len(safe_neighbors[neighbor[0]]) == 0 or (
                    len(safe_neighbors[neighbor[0]]) == 1 and safe_neighbors[neighbor[0]][0] == i):
                categories.append('rare')
            else:
                categories.append('borderline')
        elif len(neighbor) in [2, 3]:
            categories.append('borderline')
        else:
            categories.append('safe')

    instances = np.array(X_train_no_group.index.tolist())
    neighbors = instances[neighbors]
    return instances, neighbors, categories


class HybridSamplor(FOS_SMOTE):
    def __init__(self, instances: np.array, neighbors: np.array, categories: np.array, weights: dict):
        super().__init__(5, 42)
        self.instances = instances
        self.neighbors = neighbors
        self.categories = categories
        self.cat_inst_dict = dict(zip(instances, categories))
        self.oversampling_weights = weights
        self.undersampling_weights = {c: 1 - w for c, w in weights.items()}  # for now

    def undersampling(self, df: pd.DataFrame, dataset: Dataset, diff: int) -> pd.DataFrame:
        df_indices = np.array(df.index.tolist())
        df_weights = np.array([self.undersampling_weights[self.cat_inst_dict[i]] for i in df_indices])
        df_weights = df_weights / np.sum(df_weights)
        to_remove = dataset.random_state.choice(df_indices, size=diff, replace=False, p=df_weights)
        new_df = df.drop(index=to_remove)
        return new_df

    def oversampling(self, df: pd.DataFrame, dataset: Dataset, diff: int) -> pd.DataFrame:
        df_indices = np.array(df.index.tolist())
        df_weights = np.array([self.oversampling_weights[self.cat_inst_dict[i]] for i in df_indices])
        df_weights = df_weights / np.sum(df_weights)
        to_oversample = dataset.random_state.choice(df_indices, size=diff, replace=True, p=df_weights)
        chosen_to_oversample = df.loc[to_oversample]
        neighbors_to_oversample = self.neighbors[to_oversample, :]
        new_examples = self.generate_samples(chosen_to_oversample, dataset, neighbors_to_oversample)
        return pd.concat([df, new_examples])

    def generate_samples(self, df: pd.DataFrame, dataset: Dataset, neighbors: np.array) -> pd.DataFrame:
        examples = []
        for i, idx in enumerate(df.index):
            b = df.loc[[idx], :]
            neighbors_b = neighbors[i, :]
            new_example = self._generate_synthetic_example(b, dataset.train, neighbors_b, dataset)
            examples.append(new_example)
        return pd.concat(examples)