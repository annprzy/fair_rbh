import numpy as np
import pandas as pd
from distython import HEOM
from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset, query_dataset
from src.preprocessing.FOS.utils import FOS_SMOTE
from src.preprocessing.hybrid_sampling.utils import compute_classes_groups, SPIDER


def run(dataset: Dataset, eps: float = 0.5, relabel: bool = False):
    X_train, y_train = dataset.features_and_classes('train')
    X_train_no_group = X_train.loc[:, ~X_train.columns.isin([dataset.target, *dataset.sensitive])]
    cat_ord_features = [f for f, t in dataset.feature_types.items() if
                        (t == 'ordinal' or t == 'categorical') and f not in [*dataset.sensitive, dataset.target]]
    cat_ord_features = [X_train_no_group.columns.get_loc(c) for c in cat_ord_features]
    X_min_no_group = X_train_no_group[y_train == dataset.minority]
    X_maj_no_group = X_train_no_group[y_train == dataset.majority]
    metric = HEOM(X_train_no_group, cat_ord_features, nan_equivalents=[np.nan])
    dbscan = DBSCAN(eps=eps, n_jobs=-1, metric=metric.heom, min_samples=2)
    minority_clusters = dbscan.fit_predict(X_min_no_group)
    majority_clusters = dbscan.fit_predict(X_maj_no_group)

    print(f'number of minority clusters: {len(np.unique(minority_clusters))}, {np.unique(minority_clusters)}')
    print(f'number of majority clusters: {len(np.unique(majority_clusters))}, {np.unique(majority_clusters)}')

    minority_clusters_examples = X_train[y_train == dataset.minority]
    minority_clusters_examples['cluster'] = minority_clusters
    minority_clusters_examples['group'] = minority_clusters_examples[dataset.sensitive].astype(str).agg('-'.join,
                                                                                                        axis=1)
    minority_clusters_examples = minority_clusters_examples.groupby('cluster')

    majority_clusters_examples = X_train[y_train == dataset.majority]
    majority_clusters_examples['cluster'] = majority_clusters
    majority_clusters_examples['group'] = majority_clusters_examples[dataset.sensitive].astype(str).agg('-'.join,
                                                                                                        axis=1)
    majority_clusters_examples = majority_clusters_examples.groupby('cluster')

    new_majority_cluster_examples = []

    for name, cluster in majority_clusters_examples:
        new_examples_cluster = []
        k = 5
        cl = cluster.drop(columns=['cluster'])
        # metric = HEOM(X_train_no_group, cat_ord_features, nan_equivalents=[np.nan])
        # knn = NearestNeighbors(n_neighbors=k + 1, metric=metric.heom)
        # knn.fit(cluster[[c for c in cluster.columns if c not in ['group', *dataset.sensitive]]])
        unique_groups, counts = np.unique(cl['group'], return_counts=True)
        len_biggest_group = np.max(counts)
        for group in unique_groups:
            examples_group = cl[cl['group'] == group]
            to_oversample = len_biggest_group - len(examples_group)
            if to_oversample > 0:
                examples_to_oversample = examples_group.sample(n=to_oversample, replace=True, random_state=dataset.random_state)
                #examples_to_oversample = examples_to_oversample.drop(columns=['group'])
                examples_to_oversample[dataset.target] = [dataset.majority] * to_oversample
                new_examples_cluster.append(examples_to_oversample)

        cl[dataset.target] = [dataset.majority] * len(cl)
        new_majority_cluster_examples.append(pd.concat([*new_examples_cluster, cl]))

    new_minority_cluster_examples = []

    for name, cluster in minority_clusters_examples:
        new_examples_cluster = []
        k = 5
        cl = cluster.drop(columns=['cluster'])
        # metric = HEOM(X_train_no_group, cat_ord_features, nan_equivalents=[np.nan])
        # knn = NearestNeighbors(n_neighbors=k + 1, metric=metric.heom)
        # knn.fit(cluster[[c for c in cluster.columns if c not in ['group', *dataset.sensitive]]])
        unique_groups, counts = np.unique(cl['group'], return_counts=True)
        len_biggest_group = np.max(counts)
        for group in unique_groups:
            examples_group = cl[cl['group'] == group]
            to_oversample = len_biggest_group - len(examples_group)
            if to_oversample > 0:
                examples_to_oversample = examples_group.sample(n=to_oversample, replace=True, random_state=dataset.random_state)
                #examples_to_oversample = examples_to_oversample.drop(columns=['group'])
                examples_to_oversample[dataset.target] = [dataset.minority] * to_oversample
                new_examples_cluster.append(examples_to_oversample)

        cl[dataset.target] = [dataset.minority] * len(cl)
        new_minority_cluster_examples.append(pd.concat([*new_examples_cluster, cl]))

    new_examples_equal = []
    majority_clusters_sum = np.sum([len(m) for m in new_majority_cluster_examples])
    to_add_minority = majority_clusters_sum - np.sum([len(m) for m in new_minority_cluster_examples])
    print(to_add_minority, majority_clusters_sum)
    if to_add_minority > 0:
        to_add_minority = to_add_minority // len(np.unique(minority_clusters))
        for cluster in new_minority_cluster_examples:
            to_oversample = to_add_minority
            unique_groups, counts = np.unique(cluster['group'], return_counts=True)
            to_oversample = to_oversample // len(unique_groups)
            if to_oversample > 0:
                for group in unique_groups:
                    examples_group = cluster[cluster['group'] == group]
                    examples_to_oversample = examples_group.sample(n=to_oversample, replace=True,
                                                                   random_state=dataset.random_state)
                    # examples_to_oversample = examples_to_oversample.drop(columns=['group'])
                    examples_to_oversample[dataset.target] = [dataset.minority] * to_oversample
                    new_examples_equal.append(examples_to_oversample)

    # new_examples_equal = []
    # for cluster in new_majority_cluster_examples:
    #     to_oversample = majority_biggest_cluster - len(cluster)
    #     unique_groups, counts = np.unique(cluster['group'], return_counts=True)
    #     to_oversample = to_oversample // len(unique_groups)
    #     if to_oversample > 0:
    #         for group in unique_groups:
    #             examples_group = cluster[cluster['group'] == group]
    #             examples_to_oversample = examples_group.sample(n=to_oversample, replace=True,
    #                                                            random_state=dataset.random_state)
    #             # examples_to_oversample = examples_to_oversample.drop(columns=['group'])
    #             examples_to_oversample[dataset.target] = [dataset.majority] * to_oversample
    #             new_examples_equal.append(examples_to_oversample)
    #
    # biggest_cluster_scaled = majority_biggest_cluster * len(np.unique(majority_clusters)) // len(np.unique(minority_clusters))
    # print(biggest_cluster_scaled)
    # for cluster in new_minority_cluster_examples:
    #     to_oversample = biggest_cluster_scaled - len(cluster)
    #     unique_groups, counts = np.unique(cluster['group'], return_counts=True)
    #     to_oversample = to_oversample // len(unique_groups)
    #     if to_oversample > 0:
    #         for group in unique_groups:
    #             examples_group = cluster[cluster['group'] == group]
    #             examples_to_oversample = examples_group.sample(n=to_oversample, replace=True,
    #                                                            random_state=dataset.random_state)
    #             # examples_to_oversample = examples_to_oversample.drop(columns=['group'])
    #             examples_to_oversample[dataset.target] = [dataset.minority] * to_oversample
    #             new_examples_equal.append(examples_to_oversample)

    df = pd.concat([*new_examples_equal, *new_majority_cluster_examples, *new_minority_cluster_examples])
    df = df.drop(columns=['group'])
    #df = pd.concat([*new_majority_cluster_examples, *new_minority_cluster_examples])
    dataset.set_fair(df)

    # X_train, y_train = dataset.features_and_classes("train")
    # X_train_no_group = X_train.loc[:, ~X_train.columns.isin([dataset.target, *dataset.sensitive])]
    # cat_ord_features = [f for f, t in dataset.feature_types.items() if
    #                     (t == 'ordinal' or t == 'categorical') and f not in [*dataset.sensitive, dataset.target]]
    # cat_ord_features = [X_train_no_group.columns.get_loc(c) for c in cat_ord_features]
    # metric = HEOM(X_train_no_group, cat_ord_features, nan_equivalents=[np.nan])
    # knn = NearestNeighbors(n_neighbors=k + 1, metric=metric.heom)
    # knn.fit(X_train_no_group)
    #
    # spider = SPIDER(knn, k, relabel)
    #
    # new_data = []
    #
    # groups_maj_min = compute_classes_groups(dataset)
    # for group in groups_maj_min:
    #     query_basic = [f'`{key}`=={value}' if type(value) is not str else f'`{key}`=="{value}"' for key, value in
    #                    group.items() if key not in ['majority', 'minority']]
    #     query_maj = [*query_basic, f'`{dataset.target}`=={group["majority"]}']
    #     query_min = [*query_basic, f'`{dataset.target}`=={group["minority"]}']
    #     maj_subgroup = dataset.train.query(' and '.join(query_maj))
    #     min_subgroup = dataset.train.query(' and '.join(query_min))
    #
    #     undersampled = spider.undersampling(maj_subgroup, dataset)
    #
    #     minority_examples = int((len(undersampled) - len(min_subgroup)) / len(min_subgroup))
    #     minority_examples = max(1, minority_examples - 1)
    #     g = {k: v for k, v in group.items() if k not in ['majority', 'minority']}
    #     g[dataset.target] = group['minority']
    #
    #     oversampled = spider.oversampling(min_subgroup, dataset, g, minority_examples)
    #
    #     new_data.append(maj_subgroup)
    #     new_data.append(oversampled)
    #
    # new_data = pd.concat(new_data)
    # dataset.set_fair(new_data)
