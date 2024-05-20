import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters
from src.preprocessing.fawos_hybrid.utils import analyze_neighborhood, HybridSamplor


def run(dataset: Dataset, weights: dict, max_undersampling_frac: float = 0.6):
    num_groups = len([*dataset.privileged_groups, *dataset.unprivileged_groups]) * 2
    mean_subgroup_size = len(dataset.train) // num_groups

    subgroups = get_clusters(dataset)

    subgroups_len = [len(s) for _, s, _, _ in subgroups]
    max_subgroup_len = np.max(subgroups_len)
    if int(max_subgroup_len * max_undersampling_frac) > mean_subgroup_size:
        mean_subgroup_size = int(max_subgroup_len * max_undersampling_frac)

    instances, neighbors, safe_neighbors, categories = analyze_neighborhood(dataset)
    sampling = HybridSamplor(instances, neighbors, safe_neighbors, categories, weights)

    new_data = []
    for data in subgroups:
        q, subgroup, _, _ = data
        diff = len(subgroup) - mean_subgroup_size
        if diff > 0:
            new_subgroup = sampling.undersampling(subgroup, dataset, diff)
        elif diff < 0:
            new_subgroup = sampling.oversampling(subgroup, dataset, -diff)
            for k, v in q.items():
                new_subgroup[k] = [v] * len(new_subgroup)
        else:
            new_subgroup = subgroup
        new_data.append(new_subgroup)

    new_data = pd.concat(new_data)
    dataset.set_fair(new_data)
