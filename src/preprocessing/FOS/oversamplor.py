from copy import deepcopy

import pandas as pd

from src.datasets.dataset import Dataset, query_dataset
from src.preprocessing.FOS.utils import FOS_SMOTE


def run(dataset: Dataset, k: int, oversampling_factor: float = 1.0) -> None:
    priv = deepcopy(dataset.privileged_groups[0])
    unpriv = deepcopy(dataset.unprivileged_groups[0])

    fos_smote = FOS_SMOTE(k, dataset.random_state)

    D_min = query_dataset({dataset.target: dataset.minority}, dataset.train)

    priv[dataset.target] = dataset.majority
    unpriv[dataset.target] = dataset.majority
    D_pr_maj = query_dataset(priv, dataset.train)
    D_up_maj = query_dataset(unpriv, dataset.train)

    priv[dataset.target] = dataset.minority
    unpriv[dataset.target] = dataset.minority
    D_pr_min = query_dataset(priv, dataset.train)
    D_up_min = query_dataset(unpriv, dataset.train)

    n_pr_maj = len(D_pr_maj)
    n_up_maj = len(D_up_maj)
    n_pr_min = len(D_pr_min)
    n_up_min = len(D_up_min)

    s_pr = int((n_pr_maj - n_pr_min) * oversampling_factor)
    s_up = int((n_up_maj - n_up_min) * oversampling_factor)

    if s_up < s_pr:
        n_samp1 = s_up
        n_samp2 = s_pr
        D1 = D_up_min
        D2 = D_pr_min
    else:
        n_samp1 = s_pr
        n_samp2 = s_up
        D1 = D_pr_min
        D2 = D_up_min

    if n_samp1 <= len(D1):
        base1 = D1.sample(n=n_samp1, random_state=dataset.random_state)
    else:
        base1 = D1.sample(n=n_samp1, random_state=dataset.random_state, replace=True).reset_index(drop=True)
    neighbors1 = D1

    new_samples1 = fos_smote.generate_examples(base1, neighbors1, dataset, dataset.minority)

    if n_samp2 <= len(D2):
        base2 = D2.sample(n=n_samp2, random_state=dataset.random_state)
    else:
        base2 = D2.sample(n=n_samp2, random_state=dataset.random_state, replace=True).reset_index(drop=True)
    neighbors2 = D_min

    new_samples2 = fos_smote.generate_examples(base2, neighbors2, dataset, dataset.minority)

    new_train = pd.concat([dataset.train, new_samples1, new_samples2])
    dataset.set_fair(new_train)
