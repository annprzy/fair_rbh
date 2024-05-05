from __future__ import print_function, division
import random

import numpy as np
import pandas as pd
from distython import HEOM
from sklearn.neighbors import NearestNeighbors as NN

from src.datasets.dataset import Dataset


def get_ngbr(df: pd.DataFrame, knn: NN, dataset: Dataset) -> tuple:
    rand_sample_idx = dataset.random_state.integers(0, df.shape[0] - 1)
    parent_candidate = df.iloc[[rand_sample_idx]]
    ngbr = knn.kneighbors(parent_candidate, 3, return_distance=False)
    candidate_1 = df.iloc[ngbr[0][0]]
    candidate_2 = df.iloc[ngbr[0][1]]
    candidate_3 = df.iloc[ngbr[0][2]]
    parent_candidate = df.iloc[rand_sample_idx]
    return parent_candidate, candidate_2, candidate_3


def generate_samples(dataset: Dataset, df: pd.DataFrame, no_of_samples: int, cr: float = 0.8, f: float = 0.8) -> pd.DataFrame:
    new_examples = []
    cat_ord_features = [f for f, t in dataset.feature_types.items() if
                        (t == 'ordinal' or t == 'categorical')]
    cat_ord_features = [df.columns.get_loc(c) for c in cat_ord_features]
    metric = HEOM(df, cat_ord_features, nan_equivalents=[np.nan])
    knn = NN(n_neighbors=3, n_jobs=-1, metric=metric.heom).fit(df)

    for _ in range(no_of_samples):
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn, dataset)
        new_candidate = {}
        for key, value in parent_candidate.items():
            if dataset.feature_types[key] == 'categorical' and len(dataset.data[key].unique()) < 3:
                new_candidate[key] = [parent_candidate[key] if cr < dataset.random_state.random() else not parent_candidate[key]]
            elif dataset.feature_types[key] == 'categorical':
                new_candidate[key] = [dataset.random_state.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]])]
            elif dataset.feature_types[key] == 'ordinal':
                v = parent_candidate[key] if cr < dataset.random_state.random() else int(parent_candidate[key] +
                                                                                         f * (child_candidate_1[key] -
                                                                                              child_candidate_2[key]))
                new_candidate[key] = [v]
            else:
                new_candidate[key] = [abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key]))]
        new_examples.append(pd.DataFrame(new_candidate))
    new_data = pd.concat([*new_examples, df])
    return new_data