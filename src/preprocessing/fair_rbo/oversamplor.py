import numpy as np
import pandas as pd
from distython import HEOM

from sklearn.model_selection import StratifiedKFold

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters


def run(dataset: Dataset):
    rbo = FairRBO(dataset, gamma=0.05, n_steps=200, step_size=0.01, stop_probability=0.02,
              criterion='balance')
    new_data = rbo.fit_sample()
    dataset.set_fair(new_data)


def _rbf(d, eps):
    return np.exp(-(d * eps) ** 2)


def _distance(x, y, metric):
    return metric.heom(x, y).flatten()[0]
    # return np.sum(np.abs(x - y))


def _score(point, X, y, groups, current_class, current_group, epsilon, metric):
    mutual_density_score = 0.0

    for i in range(len(X)):
        rbf = _rbf(_distance(point, X[i], metric), epsilon)

        if y[i] == current_class and groups[i] == current_group:
            mutual_density_score -= rbf
        else:
            mutual_density_score += rbf

    return mutual_density_score


class FairRBO:
    def __init__(self, dataset: Dataset, gamma=0.05, n_steps=500, step_size=0.001, stop_probability=0.02,
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

    def fit_sample(self):
        cat_ord_features = [f for f, t in self.dataset.feature_types.items() if
                            (t == 'categorical') and f not in [*self.dataset.sensitive, self.dataset.target]]
        X_train, y_train = self.dataset.features_and_classes('train')
        X_full, y_full = X_train.to_numpy(), y_train.to_numpy()
        types_vector = [0 if self.dataset.feature_types[c] == 'categorical' else 1 for c in X_train.columns]
        lower_range = X_train.min(axis='columns').to_numpy().flatten().T
        upper_range = X_train.max(axis='columns').to_numpy().flatten().T
        sampling_features = [i for i, c in enumerate(X_train.columns) if c not in self.dataset.sensitive]
        cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
        metric = HEOM(X_train.to_numpy(), cat_ord_features, nan_equivalents=[np.nan])
        epsilon = 1.0 / self.gamma

        groups = X_train[self.dataset.sensitive].astype(int).astype(str).agg('-'.join, axis=1).to_list()

        clusters = get_clusters(self.dataset, self.dataset.train)
        len_clusters = [len(entry[1]) for entry in clusters]
        max_len = np.max(len_clusters)

        appended = []

        for entry in clusters:
            new_examples = []
            query, cluster, _, _ = entry
            current_group = '_'.join([str(int(query[s])) for s in self.dataset.sensitive])
            X = cluster.loc[:, cluster.columns != self.dataset.target].to_numpy()
            y = cluster[self.dataset.target].to_numpy()
            cluster = cluster.to_numpy()
            n = max_len - len(cluster)

            if n > 0:
                current_class = query[self.dataset.target]

                cluster_scores = []
                for i in range(len(cluster)):
                    cluster_point = cluster[i]
                    cluster_scores.append(_score(cluster_point, X_full, y_full, groups, current_class, current_group, epsilon, metric))

                probas = cluster_scores + np.abs(np.min(cluster_scores))
                probas /= np.sum(probas)
                while len(new_examples) < n:
                    idx = np.random.choice(range(len(cluster)), p=probas)
                    point = X[idx].copy()
                    score = cluster_scores[idx]

                    for i in range(self.n_steps):
                        if self.stop_probability is not None and self.stop_probability > self.dataset.random_state.random():
                            break
                        translation = np.zeros(len(point))
                        sign = self.dataset.random_state.choice([-1, 1])
                        random_choice_feature = self.dataset.random_state.choice(sampling_features)

                        if types_vector[random_choice_feature] == 0:
                            random_value = self.dataset.random_state.choice([i for i in range(
                                int(lower_range[random_choice_feature]), int(upper_range[random_choice_feature]) + 1) if
                                                                             i != point[random_choice_feature]])
                            translated_point = point + translation
                            translated_point[random_choice_feature] = random_value
                        else:
                            full_range = upper_range[random_choice_feature] - lower_range[random_choice_feature]
                            translation[random_choice_feature] = sign * full_range * self.step_size
                            translated_point = point + translation
                        translated_score = _score(translated_point, X_full, y_full, groups, current_class, current_group, epsilon, metric)

                        if (self.criterion == 'balance' and np.abs(translated_score) < np.abs(score)) or \
                                (self.criterion == 'minimize' and translated_score < score) or \
                                (self.criterion == 'maximize' and translated_score > score):
                            point = translated_point
                            score = translated_score

                    # print(len(new_examples), max_len)
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
