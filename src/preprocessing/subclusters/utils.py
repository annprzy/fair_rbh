from copy import deepcopy

import numpy as np
import pandas as pd
from distython import HEOM
from sklearn_extra.cluster import KMedoids, CommonNNClustering

from src.datasets.dataset import Dataset
from src.preprocessing.FOS.utils import FOS_SMOTE


def cluster_classes(df: pd.DataFrame, dataset: Dataset, metric, n_clusters: int, n_iter=300):
    # maj_group = dataset.train.loc[dataset.train[dataset.target] == dataset.majority, ~dataset.train.columns.isin([dataset.target])]
    # min_group = dataset.train.loc[dataset.train[dataset.target] == dataset.minority, ~dataset.train.columns.isin([dataset.target])]
    # n_clusters = len(dataset.privileged_groups) + len(dataset.unprivileged_groups)
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric.heom, random_state=42, max_iter=n_iter).fit(df)
    classification = kmedoids.labels_
    cluster_centers = kmedoids.cluster_centers_
    return classification, cluster_centers


def cluster_classes_eps(df: pd.DataFrame, dataset: Dataset, metric, eps: float):
    common_nn = CommonNNClustering(eps=eps, metric=metric.heom, n_jobs=-1).fit(df)
    return common_nn.labels_


def get_instances(df: pd.DataFrame, clusters: list | np.ndarray, groups: list):
    df['cluster'] = clusters
    df['group'] = groups
    grouped_clusters = []
    for name, small_df in df.groupby(['cluster'], as_index=False):
        grouped_clusters.append(
            [g.drop(columns=['cluster', 'group']) for n, g in small_df.groupby(['group'], as_index=False)])
    return grouped_clusters


def sample_subcluster(clusters: list, maj_class_center: list, min_class_center: list, max_size: int, dataset: Dataset, current_class: int, all_groups_centers: list,
                      metric) -> pd.DataFrame:
    new_data = []
    for cluster in clusters:
        whole_cluster = pd.concat(cluster)
        whole_cluster[dataset.target] = [current_class] * len(whole_cluster)
        for subcluster in cluster:
            if len(subcluster) > 2:
                subcluster = subcluster.astype(float)
                group = {s: subcluster[s].values.tolist()[0] for s in dataset.sensitive}

                _, subcluster_center = cluster_classes(subcluster, dataset, metric, 1, n_iter=1)

                #weights calculation
                # distance to majority
                instances_distances_maj = np.array(
                    [metric.heom(subcluster.loc[i, :].values.flatten().astype(float), maj_class_center[0].flatten()) for
                     i
                     in subcluster.index]).flatten()
                # distance to minority
                instances_distances_min = np.array(
                    [metric.heom(subcluster.loc[i, :].values.flatten().astype(float), min_class_center[0].flatten()) for
                     i
                     in subcluster.index]).flatten()
                instances_mean_dist = (instances_distances_maj + instances_distances_min) * 0.5
                instances_dev = (np.abs(instances_distances_maj - instances_mean_dist) + np.abs(
                    instances_distances_min - instances_mean_dist)) / 2
                instances_dev = instances_mean_dist / (instances_dev + 1)
                if current_class == dataset.majority:
                    instances_dist = instances_mean_dist / (instances_distances_maj + 1)
                else:
                    instances_dist = instances_mean_dist / (instances_distances_min + 1)
                instances_distances = instances_dev * instances_dist
                instances_distances /= np.sum(instances_distances)

                instances_distances_groups = []
                current_group_distances = None
                for entry in all_groups_centers:
                    same = True
                    k, center = entry
                    dists = np.array(
                        [metric.heom(subcluster.loc[i, :].values.flatten().astype(float), center[0].flatten()) for i
                         in subcluster.index]).flatten()
                    instances_distances_groups.append(dists)
                    for s in k.keys():
                        if k[s] != group[s]:
                            same = False
                    if same:
                        current_group_distances = deepcopy(dists)
                instances_distances_groups = np.array(instances_distances_groups)
                instances_mean_dist = np.mean(instances_distances_groups, axis=0)
                instances_dev_dist = np.abs(instances_distances_groups - instances_mean_dist)
                instances_dev_dist = np.mean(instances_dev_dist, axis=0)
                instances_dev_dist = instances_mean_dist / (instances_dev_dist + 1)

                instances_groups = instances_mean_dist / (current_group_distances + 1)
                instances_groups = instances_dev_dist * instances_groups
                instances_groups /= np.sum(instances_groups)
                instances_distances += instances_groups

                instances_distances /= np.sum(instances_distances)

                # undersampling if needed
                percentile = np.percentile(instances_distances, 80)
                percentile_needed = np.max(instances_distances) + 1
                if len(subcluster) > round(max_size * 0.8):
                    percentile_needed = max_size / len(subcluster) * 0.8 * 100
                    percentile_needed = np.percentile(instances_distances, percentile_needed)
                    subcluster_corrected = deepcopy(subcluster.iloc[np.argwhere(instances_distances < percentile_needed).flatten()])
                else:
                    subcluster_corrected = deepcopy(subcluster)
                #subcluster_corrected = deepcopy(subcluster)
                instances_distances[instances_distances >= percentile] = 0  # we still dont want them to be oversampled
                # instances_distances += 1
               #  print(instances_dev, percentile_needed, np.max(instances_distances), len(instances_distances), len(instances_distances[instances_distances < percentile_needed]), len(subcluster_corrected))

                # instances_distances = np.max(instances_distances) - instances_distances + 1
                # instances_distances /= np.sum(instances_distances)

                to_oversample = round(max_size * 0.8) - len(subcluster)

                subcluster_corrected[dataset.target] = [current_class] * len(subcluster_corrected)
                if to_oversample > 0:
                    instances_to_oversample = subcluster_corrected.sample(n=to_oversample, replace=True,
                                                                          random_state=dataset.random_state, weights=instances_distances[instances_distances < percentile_needed])
                    instances_to_oversample.reset_index(drop=True, inplace=True)
                    k = min(5, len(subcluster) - 2)

                    oversampled = FOS_SMOTE(k=k, random_state=dataset.random_state).generate_examples(
                        instances_to_oversample, subcluster_corrected, dataset, current_class, concat=False)
                    # for s in dataset.sensitive:
                    #     oversampled.loc[:, s] = [group[s]] * len(oversampled)
                    new_data.append(oversampled)
                new_data.append(subcluster_corrected)
    return pd.concat(new_data)
