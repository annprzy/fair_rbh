from abc import ABC, abstractmethod

import numpy as np
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from src.datasets.dataset import Dataset
from src.evaluation.fairness_measures import BinaryFairnessMeasures, MultiFairnessMeasures
from src.evaluation.performance_measures import BinaryPerformanceMeasures


class Classifier(ABC):
    def __init__(self, cfg_path: str):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    def fine_tune(self, dataset: Dataset, data: str):
        """fine-tune the given model
        :param dataset: the dataset used for experiments
        :param data: whether to use normal train or fair dataset
        :return: the fine-tuned model"""
        gd = GridSearchCV(estimator=self.model, param_grid=self.cfg['fine_tune'], verbose=True)
        X_train, y_train = dataset.features_and_classes(data)
        gd.fit(X_train, y_train)
        return gd

    def train(self, dataset: Dataset, data: str):
        """train the model
        :param dataset: the dataset used for experiments
        :param data: whether to use normal train or fair dataset"""
        X_train, y_train = dataset.features_and_classes(data)
        self.model.fit(X_train, y_train)

    def predict_and_evaluate(self, dataset: Dataset, fairness_type: str, calc_type: str = None) -> tuple[dict, dict]:
        """predict and evaluate
        :param dataset: the dataset used for experiments
        :param fairness_type: whether only two groups are considered (then binary) or multiple (then multi)
        :param calc_type: if fairness_type is multi, then calc_type should be either fawos or sonoda
        :return: the calculated performance and fairness measures"""
        X_test, y_test = dataset.features_and_classes("test")
        y_pred = self.model.predict(X_test)

        performance_scores = BinaryPerformanceMeasures().calculate_metrics(y_test, y_pred)
        if fairness_type == "binary":
            fairness_scores = BinaryFairnessMeasures(dataset).calculate_all(y_pred, dataset.test,
                                                                            dataset.privileged_groups[0],
                                                                            dataset.unprivileged_groups[0])
        elif fairness_type == "multi":
            fairness_scores = MultiFairnessMeasures(dataset).compute_measure('all', y_pred, dataset.test, calc_type)
        else:
            raise ValueError(f"Unknown fairness type: {fairness_type}")
        return performance_scores, fairness_scores
