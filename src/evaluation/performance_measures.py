import numpy as np
import pandas as pd
from imblearn import metrics as imbm
from sklearn import metrics as skm


class BinaryPerformanceMeasures:
    @staticmethod
    def calculate_metrics(y_true_m, y_pred_m, minority) -> dict:
        """calculate performance metrics: accuracy, balanced accuracy, F1, g-mean
        :param y_true: ground truth
        :param y_pred: predicted values
        :return: metrics"""
        y_true = y_true_m.copy()
        y_pred = y_pred_m.copy()
        if minority == 0:
            y_true[y_true_m == 0] = 1
            y_pred[y_pred_m == 0] = 1
            y_true[y_true_m == 1] = 0
            y_pred[y_pred_m == 1] = 0
        acc = skm.accuracy_score(y_true, y_pred)
        balanced_acc = skm.balanced_accuracy_score(y_true, y_pred)
        f1 = skm.f1_score(y_true, y_pred)
        gmean = imbm.geometric_mean_score(y_true, y_pred)
        return {'accuracy': acc, 'balanced_accuracy': balanced_acc, 'f1': f1, 'gmean': gmean}