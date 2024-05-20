import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset, query_dataset
from src.preprocessing.HFOS.utils import get_clusters


def sample_data(dataset: Dataset, number_of_examples: dict, save_path: str):
    new_data = []
    data = dataset.data
    clusters = get_clusters(dataset, data=data)
    print(save_path, number_of_examples, np.sum(list(number_of_examples.values())))
    for cluster in clusters:
        query, subgroup, _, _ = cluster
        group_query = '_'.join([*[str(query[s]) for s in dataset.sensitive], str(query[dataset.target])])
        number_to_sample = min(number_of_examples[group_query], len(subgroup))
        sample = subgroup.sample(number_to_sample, random_state=42, replace=False)
        new_data.append(sample)
    new_data = pd.concat(new_data)
    new_data = new_data.reset_index(drop=True)
    new_data.to_csv(save_path, index=False)


def get_number_samples(number_examples: int, dataset: Dataset, group_imbalance: dict | None = None,
                       class_imbalance: dict | None = None,
                       group_class_imbalance: dict | None = None):
    numbers = {}
    data = dataset.data
    if group_imbalance is not None and class_imbalance is None:
        sum_values = np.sum(list(group_imbalance.values()))
        numbers_groups = {g: number_examples * i / sum_values for g, i in group_imbalance.items()}

        for group in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
            group_key = '_'.join([str(k) for k in group.values()])
            query0 = {**group, dataset.target: dataset.minority}
            len_query0 = len(query_dataset(query0, data))
            key0 = '_'.join([str(k) for k in query0.values()])
            query1 = {**group, dataset.target: dataset.majority}
            len_query1 = len(query_dataset(query1, data))
            key1 = '_'.join([str(k) for k in query1.values()])
            c0 = len_query0 / (len_query0 + len_query1)
            c1 = len_query1 / (len_query0 + len_query1)
            numbers[key0] = round(numbers_groups[group_key] * c0)
            numbers[key1] = round(numbers_groups[group_key] * c1)

    elif group_imbalance is None and class_imbalance is not None:
        sum_values = np.sum(list(class_imbalance.values()))
        numbers_classes = {g: number_examples * i / sum_values for g, i in class_imbalance.items()}
        for c in [dataset.majority, dataset.minority]:
            len_class = len(query_dataset({dataset.target: c}, data))
            for group in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
                query0 = {**group, dataset.target: c}
                key0 = '_'.join([str(k) for k in query0.values()])
                len_query0 = len(query_dataset(query0, data))
                numbers[key0] = round(numbers_classes[str(c)] * len_query0 / len_class)

    elif group_imbalance is not None and class_imbalance is not None:
        # if both are passes we assume that the given class ratios should be present in each group (so each group has
        # the same proportion of classes)
        sum_values = np.sum(list(class_imbalance.values()))
        frac_classes = {g: i / sum_values for g, i in class_imbalance.items()}
        sum_values = np.sum(list(group_imbalance.values()))
        numbers_groups = {g: number_examples * i / sum_values for g, i in group_imbalance.items()}
        for g in numbers_groups.keys():
            for c in frac_classes.keys():
                key0 = '_'.join([str(g), str(c)])
                numbers[key0] = round(numbers_groups[g] * frac_classes[c])

    if group_class_imbalance is not None:
        sum_values = np.sum(list(group_class_imbalance.values()))
        frac = {g: i / sum_values for g, i in group_class_imbalance.items()}
        numbers = {g: round(number_examples * i) for g, i in frac.items()}

    return numbers
