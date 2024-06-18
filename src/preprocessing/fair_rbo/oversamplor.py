import numpy as np
import pandas as pd
import seaborn as sns
from distython import HEOM, HVDM
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters


def run(dataset: Dataset):
    rbo = FairRBO(dataset, gamma=0.01, n_steps=200, step_size=0.001, stop_probability=0.02,
              criterion='balance')
    new_data = rbo.fit_sample()
    dataset.set_fair(new_data)


def _rbf(d, eps):
    return np.exp(-(d * eps) ** 2)


def _distance(x, y, metric):
    #return metric.hvdm(x, y).flatten()[0]
    return metric.heom(x[:-1], y[:-1]).flatten()[0]
    # return np.sum(np.abs(x - y))


def _score(point, X, y, groups, current_class, current_group, epsilon, metric, mapping, return_distances=False):
    mutual_density_score = 0.0
    distances = []

    for i in range(len(X)):
        group_class_x = '_'.join([groups[i], str(int(y[i]))])
        dist = _distance(point, np.append(X[i], mapping[group_class_x]), metric)
        rbf = _rbf(dist, epsilon)

        if y[i] == current_class and groups[i] == current_group:
            mutual_density_score -= rbf
            distances.append(dist)
        # elif y[i] == current_class:
        #     mutual_density_score -= rbf * 0.2
            #distances.append(dist)
        else:
            mutual_density_score += rbf
    if return_distances:
        return mutual_density_score, distances
    return mutual_density_score


class FairRBO:
    def __init__(self, dataset: Dataset, k=2, gamma=0.06, n_steps=100, step_size=0.001, stop_probability=0.02,
                 criterion='balance'):
        assert criterion in ['balance', 'minimize', 'maximize']
        assert 0.0 <= stop_probability <= 1.0
        self.dataset = dataset
        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.minority_class = dataset.minority
        self.k = k

    def fit_sample(self):
        cat_ord_features = [f for f, t in self.dataset.feature_types.items() if
                            (t == 'categorical') and f not in [self.dataset.target]]
        X_train, y_train = self.dataset.features_and_classes('train')
        X_full, y_full = X_train.to_numpy(), y_train.to_numpy()
        types_vector = [0 if self.dataset.feature_types[c] == 'categorical' else 1 for c in X_train.columns]
        lower_range = X_train.min(axis='columns').to_numpy().flatten().T
        upper_range = X_train.max(axis='columns').to_numpy().flatten().T
        sampling_features = [i for i, c in enumerate(X_train.columns) if c not in self.dataset.sensitive]
        #sampling_features = [i for i, c in enumerate(X_train.columns) if types_vector[i] != 0 and c not in self.dataset.sensitive]
        cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
        group_class = self.dataset.train[[*self.dataset.sensitive, self.dataset.target]].astype(int).astype(str).agg('_'.join, axis=1)
        unique_groups = np.unique(group_class)
        mapping = {g: i for i, g in enumerate(unique_groups)}
        group_class_m = pd.DataFrame(np.array([[mapping[g] for g in group_class]]).T, columns=['group_class'])
        metric = HVDM(pd.concat([X_train, group_class_m], axis=1).to_numpy(), [X_train.shape[1]], cat_ord_features, nan_equivalents=[np.nan], normalised='')
        metric = HEOM(X_train.to_numpy(), cat_ord_features, nan_equivalents=[np.nan], normalised='')
        epsilon = 1.0 / self.gamma

        #knn = NearestNeighbors(n_neighbors=self.k+1, algorithm='ball_tree', metric=metric.hvdm, n_jobs=-1).fit(pd.concat([X_train, group_class_m], axis=1).to_numpy())

        groups = np.array(X_train[self.dataset.sensitive].astype(int).astype(str).agg('-'.join, axis=1).to_list())

        clusters = get_clusters(self.dataset, self.dataset.train)
        len_clusters = [len(entry[1]) for entry in clusters]
        max_len = np.max(len_clusters)

        appended = []

        for entry in clusters:
            new_examples = []
            query, cluster_pd, _, _ = entry
            cluster_vis = cluster_pd.loc[:, cluster_pd.columns != self.dataset.target]
            current_group = '_'.join([str(int(query[s])) for s in self.dataset.sensitive])
            X = cluster_pd.loc[:, cluster_pd.columns != self.dataset.target].to_numpy()
            y = cluster_pd[self.dataset.target].to_numpy()
            cluster = cluster_pd.to_numpy()
            n = max_len - len(cluster)

            current_class = query[self.dataset.target]
            group_class_current = '_'.join([current_group, str(int(current_class))])
            group_class_current = mapping[group_class_current]

            if n > 0:
                cluster_scores = []
                cluster_weights = []
                for i in range(len(cluster)):
                    cluster_point = cluster[i]
                    c_score, c_dists = _score(np.append(cluster_point[:-1], group_class_current), X_full, y_full, groups, current_class, current_group, epsilon, metric, mapping, return_distances=True)
                    cluster_scores.append(c_score)
                    mean_knn_dist = np.mean(sorted(c_dists[:self.k]))
                    cluster_weights.append(mean_knn_dist)
                # print('_'.join([current_group, str(int(current_class))]))
                # if '_'.join([current_group, str(int(current_class))]) == '0_1':
                #     print(cluster_scores)
                #     pca = PCA(n_components=2, random_state=42)
                #     cat_ord_train = self.dataset.perform_encoding('cont_ord_cat', X_train, X_train)
                #     cat_ord_clust = self.dataset.perform_encoding('cont_ord_cat', X_train, cluster_vis)
                #     pca.fit(cat_ord_train)
                #     pca_features = pca.transform(cat_ord_clust)
                #     PC1 = pca_features[:, 0]
                #     PC2 = pca_features[:, 1]
                #     df = pd.DataFrame({'PC1': PC1, 'PC2': PC2, 'score': cluster_scores})
                #     fig = sns.scatterplot(df, x='PC1', y='PC2', hue='score').get_figure()
                #     fig.savefig("out.png")
                cluster_weights = np.array(cluster_weights)
                cluster_weights = np.max(cluster_weights) + 1 - cluster_weights
                cluster_weights /= np.sum(cluster_weights)
                while len(new_examples) < n:
                    idx = self.dataset.random_state.choice(range(len(cluster)), p=cluster_weights)
                    point = X[idx].copy()
                    score = cluster_scores[idx]

                    for i in range(self.n_steps):
                        if self.stop_probability is not None and self.stop_probability > self.dataset.random_state.random():
                            break
                        translation = np.zeros(len(point))
                        sign = self.dataset.random_state.choice([-1, 1])
                        random_choice_feature = self.dataset.random_state.choice(sampling_features)

                        if types_vector[random_choice_feature] == 0:
                            unique_rc, counts_rc = np.unique(X_train.loc[y_train == current_class, X_train.columns[random_choice_feature]], return_counts=True)
                            counts_rc = counts_rc.astype(float) / np.sum(counts_rc)
                            random_value = self.dataset.random_state.choice(unique_rc, p=counts_rc)
                            translated_point = point + translation
                            translated_point[random_choice_feature] = random_value
                        else:
                            full_range = upper_range[random_choice_feature] - lower_range[random_choice_feature]
                            translation[random_choice_feature] = sign * full_range * self.step_size
                            translated_point = point + translation
                        translated_score = _score(np.append(translated_point, group_class_current), X_full, y_full, groups, current_class, current_group, epsilon, metric, mapping)

                        if (self.criterion == 'balance' and np.abs(translated_score) < np.abs(score)) or \
                                (self.criterion == 'minimize' and translated_score < score) or \
                                (self.criterion == 'maximize' and translated_score > score):
                            point = translated_point
                            score = translated_score

                    #print(len(new_examples), max_len)
                    new_examples.append(point)
                new_examples = np.array(new_examples)
                new_X = np.concatenate([X, new_examples])
                new_y = np.concatenate([y, current_class * np.ones(len(new_examples))])

                new_data = np.c_[new_X, new_y]
            else:
                new_data = np.c_[X, y]
            appended.append(new_data)
        new_data_full = np.concatenate(appended)
        new_data_full = pd.DataFrame(new_data_full, columns=self.dataset.train.columns)

        return new_data_full
