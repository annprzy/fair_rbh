import numpy as np
import pandas as pd
from distython import HEOM

from sklearn.model_selection import StratifiedKFold

from src.datasets.dataset import Dataset


def run(dataset: Dataset):
    rbo = RBO(dataset, gamma=0.05, n_steps=200, step_size=0.01, stop_probability=0.02,
              criterion='balance', n=None)
    X_train, y_train = dataset.features_and_classes('train')
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    new_data = rbo.fit_sample(X_train, y_train)
    dataset.set_fair(new_data)


def _rbf(d, eps):
    return np.exp(-(d * eps) ** 2)


def _distance(x, y, metric):
    return metric.heom(x, y).flatten()[0]
    # return np.sum(np.abs(x - y))


def _score(point, X, y, minority_class, epsilon, metric):
    mutual_density_score = 0.0

    for i in range(len(X)):
        rbf = _rbf(_distance(point, X[i], metric), epsilon)

        if y[i] == minority_class:
            mutual_density_score -= rbf
        else:
            mutual_density_score += rbf

    return mutual_density_score


class RBO:
    def __init__(self, dataset: Dataset, gamma=0.1, n_steps=200, step_size=0.01, stop_probability=0.02,
                 criterion='balance',
                 n=None):
        assert criterion in ['balance', 'minimize', 'maximize']
        assert 0.0 <= stop_probability <= 1.0
        self.dataset = dataset
        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.minority_class = dataset.minority
        self.n = n

    def fit_sample(self, X, y):
        cat_ord_features = [f for f, t in self.dataset.feature_types.items() if
                            (t == 'categorical') and f not in [*self.dataset.sensitive, self.dataset.target]]
        X_train, y_train = self.dataset.features_and_classes('train')
        types_vector = [0 if self.dataset.feature_types[c] == 'categorical' else 1 for c in X_train.columns]
        lower_range = X_train.min(axis='columns').to_numpy().flatten().T
        upper_range = X_train.max(axis='columns').to_numpy().flatten().T
        cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
        metric = HEOM(X_train.to_numpy(), cat_ord_features, nan_equivalents=[np.nan])
        epsilon = 1.0 / self.gamma
        classes = np.unique(y)

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        minority_points = X[y == minority_class]

        if self.n is None:
            n = sum(y != minority_class) - sum(y == minority_class)
        else:
            n = self.n

        if n == 0:
            return X, y

        minority_scores = []

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            minority_scores.append(_score(minority_point, X, y, minority_class, epsilon, metric))

        probas = minority_scores + np.abs(np.min(minority_scores))
        probas /= np.sum(probas)

        appended = []

        while len(appended) < n:
            idx = self.dataset.random_state.choice(range(len(minority_points)), p=probas)
            point = minority_points[idx].copy()
            score = minority_scores[idx]

            for i in range(self.n_steps):
                if self.stop_probability is not None and self.stop_probability > self.dataset.random_state.random():
                    break

                translation = np.zeros(len(point))
                sign = self.dataset.random_state.choice([-1, 1])
                random_choice_feature = self.dataset.random_state.choice(range(len(point)))
                if types_vector[random_choice_feature] == 0:
                    random_value = self.dataset.random_state.choice(
                        [i for i in range(int(lower_range[random_choice_feature]), int(upper_range[random_choice_feature]) + 1) if
                         i != point[random_choice_feature]])
                    translated_point = point + translation
                    translated_point[random_choice_feature] = random_value
                else:
                    full_range = upper_range[random_choice_feature] - lower_range[random_choice_feature]
                    translation[random_choice_feature] = sign * full_range * self.step_size
                    translated_point = point + translation
                translated_score = _score(translated_point, X, y, minority_class, epsilon, metric)

                if (self.criterion == 'balance' and np.abs(translated_score) < np.abs(score)) or \
                        (self.criterion == 'minimize' and translated_score < score) or \
                        (self.criterion == 'maximize' and translated_score > score):
                    point = translated_point
                    score = translated_score

            appended.append(point)

        new_X = np.concatenate([X, appended])
        new_y = np.concatenate([y, minority_class * np.ones(len(appended))])

        new_data = np.c_[new_X, new_y]
        new_data = pd.DataFrame(new_data, columns=self.dataset.train.columns)
        return new_data

        #return np.concatenate([X, appended]), np.concatenate([y, minority_class * np.ones(len(appended))])


class RBOSelection:
    def __init__(self, dataset: Dataset, classifier, measure, n_splits=5, gammas=(0.05,), n_steps=500, step_size=0.001,
                 stop_probability=0.02, criterion='balance', n=None):
        self.classifier = classifier
        self.measure = measure
        self.n_splits = n_splits
        self.gammas = gammas
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.minority_class = dataset.minority
        self.dataset = dataset
        self.n = n
        self.selected_gamma = None
        self.skf = StratifiedKFold(n_splits=n_splits)

    def fit_sample(self):
        X, y = self.dataset.features_and_classes('train')
        X = X.to_numpy()
        y = y.to_numpy()
        self.skf.get_n_splits(X, y)

        best_score = -np.inf

        for gamma in self.gammas:
            scores = []

            for train_idx, test_idx in self.skf.split(X, y):
                X_train, y_train = RBO(self.dataset, gamma=gamma, n_steps=self.n_steps, step_size=self.step_size,
                                       stop_probability=self.stop_probability, criterion=self.criterion,
                                       n=self.n).fit_sample(X[train_idx], y[train_idx])

                classifier = self.classifier.fit(X_train, y_train)
                predictions = classifier.predict(X[test_idx])
                scores.append(self.measure(y[test_idx], predictions))

            score = np.mean(scores)

            if score > best_score:
                self.selected_gamma = gamma

                best_score = score

        return RBO(self.dataset, gamma=self.selected_gamma, n_steps=self.n_steps, step_size=self.step_size,
                   stop_probability=self.stop_probability, criterion=self.criterion,
                   n=self.n).fit_sample(X, y)
