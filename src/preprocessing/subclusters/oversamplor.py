import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from distython import HEOM

from src.datasets.dataset import Dataset
# from src.experiments import init_dataset
from src.preprocessing.subclusters.utils import cluster_classes, get_instances, sample_subcluster, \
    cluster_classes_eps


def run(dataset: Dataset, n_clusters):
    X_train, y_train = dataset.features_and_classes('train')

    majority_data = dataset.train.loc[dataset.train[dataset.target] == dataset.majority, ~dataset.train.columns.isin([dataset.target])]
    majority_groups = majority_data[dataset.sensitive].astype(str).agg('-'.join, axis=1)
    minority_data = dataset.train.loc[dataset.train[dataset.target] == dataset.minority, ~dataset.train.columns.isin([dataset.target])]
    minority_groups = minority_data[dataset.sensitive].astype(str).agg('-'.join, axis=1)

    cat_ord_features = [f for f, t in dataset.feature_types.items() if
                        (t == 'categorical') and f not in [dataset.target]]
    cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
    metric = HEOM(X_train, cat_ord_features, nan_equivalents=[np.nan])

    all_groups = dataset.train[dataset.sensitive].astype(str).agg('-'.join, axis=1)
    all_groups_instances = get_instances(X_train, [0] * len(all_groups), all_groups)[0]
    all_groups_dicts = [{s: g[s].values.tolist()[0] for s in dataset.sensitive} for g in all_groups_instances]
    all_groups_centers = []
    for k, g in zip(all_groups_dicts, all_groups_instances):
        _, center = cluster_classes(g, dataset, metric, n_clusters=1, n_iter=1)
        all_groups_centers.append([k, center])

    clusters_minority, _ = cluster_classes(minority_data, dataset, metric, n_clusters=n_clusters)
    clusters_majority, _ = cluster_classes(majority_data, dataset, metric, n_clusters=n_clusters)
    # print(np.unique(clusters_majority, return_counts=True), np.unique(clusters_majority, return_counts=True))

    _, majority_center = cluster_classes(majority_data, dataset, metric, 1, n_iter=1)
    _, minority_center = cluster_classes(minority_data, dataset, metric, 1, n_iter=1)

    # min_cluster_instances = get_instances(minority_data, clusters_minority, minority_groups)
    # maj_cluster_instances = get_instances(majority_data, clusters_majority, majority_groups)
    min_cluster_instances = get_instances(minority_data, [1] * len(minority_data), minority_groups)
    maj_cluster_instances = get_instances(majority_data, [1] * len(majority_data), majority_groups)

    lens_min_subclusters = [len(i) for j in min_cluster_instances for i in j]
    lens_maj_subclusters = [len(i) for j in maj_cluster_instances for i in j]

    max_size = np.max([*lens_min_subclusters, *lens_maj_subclusters])

    oversampled_instances_minority = sample_subcluster(min_cluster_instances, majority_center, minority_center, max_size, dataset, dataset.minority, all_groups_centers, metric)
    oversampled_instances_majority = sample_subcluster(maj_cluster_instances, majority_center, minority_center, max_size, dataset, dataset.majority, all_groups_centers, metric)

    new_data = pd.concat([oversampled_instances_majority, oversampled_instances_minority]).reset_index(drop=True)
    dataset.set_fair(new_data)
