from copy import deepcopy

import pandas as pd

from src.datasets.dataset import Dataset, query_dataset
from src.preprocessing.FOS.utils import FOS_SMOTE


def run(dataset: Dataset, k: int, oversampling_factor: float = 1.0, distance_type='heom') -> None:
    priv = deepcopy(dataset.privileged_groups[0])
    unpriv = deepcopy(dataset.unprivileged_groups[0])

    fos_smote = FOS_SMOTE(k, dataset.random_state, distance_type=distance_type)

    D_min = query_dataset({dataset.target: dataset.minority}, dataset.train)
    D_maj = query_dataset({dataset.target: dataset.majority}, dataset.train)

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

    if n_pr_maj >= n_pr_min:
        s_pr = int((n_pr_maj - n_pr_min) * oversampling_factor)
        D_pr = D_pr_min
        neighbors_pr = [D_pr, D_min]
    else:
        s_pr = int((n_pr_min - n_pr_maj) * oversampling_factor)
        D_pr = D_pr_maj
        neighbors_pr = [D_pr, D_maj]
    if n_up_maj >= n_up_min:
        s_up = int((n_up_maj - n_up_min) * oversampling_factor)
        D_up = D_up_min
        neighbors_up = [D_up, D_min]
    else:
        s_up = int((n_up_min - n_up_maj) * oversampling_factor)
        D_up = D_up_maj
        neighbors_up = [D_up, D_maj]

    if s_up < s_pr:
        n_samp1 = s_up
        n_samp2 = s_pr
        D1 = D_up
        D2 = D_pr
        neighbors1 = neighbors_up[0]
        neighbors2 = neighbors_pr[1]
    else:
        n_samp1 = s_pr
        n_samp2 = s_up
        D1 = D_pr
        D2 = D_up
        neighbors1 = neighbors_pr[0]
        neighbors2 = neighbors_up[1]

    new_samples1, new_samples2 = None, None
    if n_samp1 > 0:
        if n_samp1 <= len(D1):
            base1 = D1.sample(n=n_samp1, random_state=dataset.random_state)
        else:
            base1 = D1.sample(n=n_samp1, random_state=dataset.random_state, replace=True).reset_index(drop=True)

        new_samples1 = fos_smote.generate_examples(base1, neighbors1, dataset, dataset.minority)

    if n_samp2 > 0:
        if n_samp2 <= len(D2):
            base2 = D2.sample(n=n_samp2, random_state=dataset.random_state)
        else:
            base2 = D2.sample(n=n_samp2, random_state=dataset.random_state, replace=True).reset_index(drop=True)

        new_samples2 = fos_smote.generate_examples(base2, neighbors2, dataset, dataset.minority)

    if new_samples1 is not None and new_samples2 is not None:
        new_train = pd.concat([dataset.train, new_samples1, new_samples2])
    elif new_samples1 is not None:
        new_train = pd.concat([dataset.train, new_samples1])
    elif new_samples2 is not None:
        new_train = pd.concat([dataset.train, new_samples2])
    else:
        new_train = deepcopy(dataset.train)

    dataset.set_fair(new_train)
