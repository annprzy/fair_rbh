from copy import deepcopy

import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset, query_dataset


class BinaryFairnessMeasures:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @staticmethod
    def _query_dataset(query: dict, test_set: pd.DataFrame):
        result = query_dataset(query, test_set)
        return len(result)

    def _true_positives(self, test_set: pd.DataFrame, group: dict):
        group_dict = deepcopy(group)
        group_dict['y_pred'] = self.dataset.privileged_class
        group_dict[self.dataset.target] = self.dataset.privileged_class
        return self._query_dataset(group_dict, test_set)

    def _true_negatives(self, test_set: pd.DataFrame, group: dict):
        group_dict = deepcopy(group)
        group_dict['y_pred'] = self.dataset.unprivileged_class
        group_dict[self.dataset.target] = self.dataset.unprivileged_class
        return self._query_dataset(group_dict, test_set)

    def _false_negatives(self, test_set: pd.DataFrame, group: dict):
        group_dict = deepcopy(group)
        group_dict['y_pred'] = self.dataset.unprivileged_class
        group_dict[self.dataset.target] = self.dataset.privileged_class
        return self._query_dataset(group_dict, test_set)

    def _false_positives(self, test_set: pd.DataFrame, group: dict):
        group_dict = deepcopy(group)
        group_dict['y_pred'] = self.dataset.privileged_class
        group_dict[self.dataset.target] = self.dataset.unprivileged_class
        return self._query_dataset(group_dict, test_set)

    def _confusion_matrix(self, test_set: pd.DataFrame, group: dict):
        return self._true_positives(test_set, group), self._true_negatives(test_set, group), self._false_negatives(
            test_set, group), self._false_positives(test_set, group)

    def statistical_parity(self, y_pred, test: pd.DataFrame, priv: dict, unpriv: dict) -> float:
        test_set = deepcopy(test)
        test_set['y_pred'] = y_pred

        if unpriv is None:
            priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
            sp_priv = (priv_tp + priv_fp) / (priv_tp + priv_tn + priv_fn + priv_fp)
            return sp_priv
        priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
        sp_priv = (priv_tp + priv_fp) / (priv_tp + priv_tn + priv_fn + priv_fp)
        unpriv_tp, unpriv_tn, unpriv_fn, unpriv_fp = self._confusion_matrix(test_set, unpriv)
        sp_unpriv = (unpriv_tp + unpriv_fp) / (unpriv_tp + unpriv_tn + unpriv_fn + unpriv_fp)
        sp = sp_priv - sp_unpriv
        return sp

    def accuracy_parity(self, y_pred, test:pd.DataFrame, priv: dict, unpriv: dict) -> float:
        test_set = deepcopy(test)
        test_set['y_pred'] = y_pred

        if unpriv is None:
            priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
            acc_priv = (priv_tp + priv_tn) / (priv_tp + priv_tn + priv_fn + priv_fp)
            return acc_priv

        priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
        unpriv_tp, unpriv_tn, unpriv_fn, unpriv_fp = self._confusion_matrix(test_set, unpriv)
        acc_priv = (priv_tp + priv_tn) / (priv_tp + priv_tn + priv_fn + priv_fp)
        acc_unpriv = (unpriv_tp + unpriv_tn) / (unpriv_tp + unpriv_tn + unpriv_fn + unpriv_fp)
        acc = acc_priv - acc_unpriv
        return acc

    def equal_opportunity(self, y_pred, test: pd.DataFrame, priv: dict, unpriv: dict) -> float:
        test_set = deepcopy(test)
        test_set['y_pred'] = y_pred

        if unpriv is None:
            priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
            try:
                eo_priv = priv_fn / (priv_fn + priv_tp)
                return eo_priv
            except ZeroDivisionError:
                return np.inf

        priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
        unpriv_tp, unpriv_tn, unpriv_fn, unpriv_fp = self._confusion_matrix(test_set, unpriv)
        try:
            eo_priv = priv_fn / (priv_fn + priv_tp)
            eo_unpriv = unpriv_fn / (unpriv_fn + unpriv_tp)
            eo = eo_priv - eo_unpriv
        except ZeroDivisionError:
            eo = np.inf
        return eo

    def average_odds(self, y_pred, test: pd.DataFrame, priv: dict, unpriv: dict) -> float:
        test_set = deepcopy(test)
        test_set['y_pred'] = y_pred

        if unpriv is None:
            priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
            try:
                x1_priv = priv_fp / (priv_fp + priv_tn)
                x2_priv = priv_tp / (priv_tp + priv_fn)
                return x1_priv, x2_priv
            except ZeroDivisionError:
                return np.inf, np.inf

        priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
        unpriv_tp, unpriv_tn, unpriv_fn, unpriv_fp = self._confusion_matrix(test_set, unpriv)
        try:
            x1_priv = priv_fp / (priv_fp + priv_tn)
            x1_unpriv = unpriv_fp / (unpriv_fp + unpriv_tn)
            x1 = x1_priv - x1_unpriv

            x2_priv = priv_tp / (priv_tp + priv_fn)
            x2_unpriv = unpriv_tp / (unpriv_tp + unpriv_fn)
            x2 = x2_priv - x2_unpriv

            ao = 0.5 * x1 + 0.5 * x2
        except ZeroDivisionError:
            ao = np.inf
        return ao

    def average_absolute_odds(self, y_pred, test: pd.DataFrame, priv: dict, unpriv: dict) -> float:
        test_set = deepcopy(test)
        test_set['y_pred'] = y_pred

        if unpriv is None:
            priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
            try:
                x1_priv = priv_fp / (priv_fp + priv_tn)
                x2_priv = priv_tp / (priv_tp + priv_fn)
                return x1_priv, x2_priv
            except ZeroDivisionError:
                return np.inf, np.inf

        priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
        unpriv_tp, unpriv_tn, unpriv_fn, unpriv_fp = self._confusion_matrix(test_set, unpriv)
        try: 
            x1_priv = priv_fp / (priv_fp + priv_tn)
            x1_unpriv = unpriv_fp / (unpriv_fp + unpriv_tn)
            x1 = abs(x1_priv - x1_unpriv)

            x2_priv = priv_tp / (priv_tp + priv_fn)
            x2_unpriv = unpriv_tp / (unpriv_tp + unpriv_fn)
            x2 = abs(x2_priv - x2_unpriv)

            ao = 0.5 * x1 + 0.5 * x2
        except ZeroDivisionError:
            ao = np.inf
        return ao

    def disparate_impact(self, y_pred, test: pd.DataFrame, priv: dict, unpriv: dict) -> float:
        test_set = deepcopy(test)
        test_set['y_pred'] = y_pred

        if unpriv is None:
            priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
            di_priv = (priv_tp + priv_fp) / (priv_tp + priv_tn + priv_fn + priv_fp)
            return di_priv
        priv_tp, priv_tn, priv_fn, priv_fp = self._confusion_matrix(test_set, priv)
        unpriv_tp, unpriv_tn, unpriv_fn, unpriv_fp = self._confusion_matrix(test_set, unpriv)
        di_priv = (priv_tp + priv_fp) / (priv_tp + priv_tn + priv_fn + priv_fp)
        di_unpriv = (unpriv_tp + unpriv_fp) / (unpriv_tp + unpriv_tn + unpriv_fn + unpriv_fp)
        di = di_unpriv / di_priv if di_priv != 0 else np.nan
        return di

    def adapted_disparate_impact(self, y_pred, test: pd.DataFrame, priv: dict, unpriv: dict):
        if unpriv is None:
            return self.disparate_impact(y_pred, test, priv, unpriv)
        di = self.disparate_impact(y_pred, test, priv, unpriv)
        if di > 1.0:
            di = 1 / di
        return di

    def calculate_all(self, y_pred, test_set: pd.DataFrame, priv: dict, unpriv: dict):
        sp = self.statistical_parity(y_pred, test_set, priv, unpriv)
        acc = self.accuracy_parity(y_pred, test_set, priv, unpriv)
        eo = self.equal_opportunity(y_pred, test_set, priv, unpriv)
        ao = self.average_odds(y_pred, test_set, priv, unpriv)
        aao = self.average_absolute_odds(y_pred, test_set, priv, unpriv)
        di = self.disparate_impact(y_pred, test_set, priv, unpriv)
        adi = self.adapted_disparate_impact(y_pred, test_set, priv, unpriv)
        return sp, acc, eo, ao, aao, di, adi

    def compute_dict(self, y_pred, test_set: pd.DataFrame, priv: dict, unpriv: dict):
        measures = self.calculate_all(y_pred, test_set, priv, unpriv)
        result = {}
        for m, n in zip(measures, ['statistical_parity', 'accuracy', 'equal_opportunity', 'average_odds', 'average_absolute_odds',
                                   'disparate_impact', 'adapted_disparate_impact']):
            result[n] = m
        return result


class MultiFairnessMeasures:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.binary_measures = BinaryFairnessMeasures(dataset)
        self.measures_dict = {
            'statistical_parity': self.binary_measures.statistical_parity,
            'accuracy': self.binary_measures.accuracy_parity,
            'equal_opportunity': self.binary_measures.equal_opportunity,
            'average_odds': self.binary_measures.average_odds,
            'average_absolute_odds': self.binary_measures.average_absolute_odds,
            'disparate_impact': self.binary_measures.disparate_impact,
            'adapted_disparate_impact': self.binary_measures.adapted_disparate_impact,
            'all': self.binary_measures.calculate_all,
        }

    def compute_measure(self, measure: str, y_pred, test_set: pd.DataFrame, calc_type: str):

        # results_per_attr = self.compute_per_attribute(measure, y_pred, test_set)
        # results_per_group = self.compute_per_group(measure, y_pred, test_set)

        if measure == 'all':
            measures = ['statistical_parity', 'accuracy', 'equal_opportunity', 'average_odds', 'average_absolute_odds',
                        'disparate_impact']
        else:
            measures = [measure]
        results_dict = {}
        for i, m in enumerate(measures):
            if calc_type == 'fawos':
                results = self.compute_fawos(m, y_pred, test_set)
            elif calc_type == 'sonoda':
                results = self.compute_sonoda(m, y_pred, test_set)
            else:
                calc_type = 'default'
                results = self.compute_default(m, y_pred, test_set)
            results_dict[f'{m}'] = results
            # for r, attr in zip(results_per_attr.T[i], self.dataset.sensitive):
            #     results_dict[f'{m}_{attr}'] = r
            # for r, attr in zip(results_per_group.T[i],
            #                    [*self.dataset.privileged_groups, *self.dataset.unprivileged_groups]):
            #     results_dict[f'{m}_{attr}'] = r

        return results_dict

    def compute_per_attribute(self, measure: str, y_pred, test_set: pd.DataFrame):
        results = []
        priv, unpriv = self.dataset.compute_privileged_unprivileged_values(self.dataset.sensitive, self.dataset.train)
        for attr, p, up in zip(self.dataset.sensitive, priv, unpriv):
            p_dict = {attr: p[0]}
            up_dict = {attr: up[0]}
            result = self.measures_dict[measure](y_pred, test_set, p_dict, up_dict)
            results.append(result)
        results = np.array(results)
        return results

    def compute_per_group(self, measure: str, y_pred, test_set: pd.DataFrame):
        results = []
        for priv in self.dataset.privileged_groups:
            result = self.measures_dict[measure](y_pred, test_set, {}, priv)
            results.append(result)
        for unpriv in self.dataset.unprivileged_groups:
            result = self.measures_dict[measure](y_pred, test_set, {}, unpriv)
            results.append(result)
        results = np.array(results)
        return results

    def compute_single_score_group(self, measure: str, y_pred, test_set: pd.DataFrame):
        results = []
        for priv in self.dataset.privileged_groups:
            result = self.measures_dict[measure](y_pred, test_set, priv, None)
            results.append(result)
        for unpriv in self.dataset.unprivileged_groups:
            result = self.measures_dict[measure](y_pred, test_set, unpriv, None)
            results.append(result)
        return results

    def compute_fawos(self, measure: str, y_pred, test_set: pd.DataFrame):
        results = self.compute_per_attribute(measure, y_pred, test_set)
        for priv in self.dataset.privileged_groups:
            for unpriv in self.dataset.unprivileged_groups:
                result = self.measures_dict[measure](y_pred, test_set, priv, unpriv)
                results.append(result)
        results = np.array(results)
        return results.mean(axis=0)

    def compute_sonoda(self, measure: str, y_pred, test_set: pd.DataFrame):
        results = self.compute_per_group(measure, y_pred, test_set)
        min_result = np.min(results, axis=0)
        max_result = np.max(results, axis=0)
        return np.abs(max_result - min_result)

    def compute_default(self, measure: str, y_pred, test_set: pd.DataFrame):
        if measure not in ['average_odds', 'average_absolute_odds', 'disparate_impact']:
            results = np.array(self.compute_single_score_group(measure, y_pred, test_set))
            if np.inf in results:
                return np.inf
            results = np.max(results) - results
            len_results = len(results) - 1
            result = np.sum(results) / len_results
            return result
        elif measure == 'disparate_impact':
            results = np.array(self.compute_single_score_group(measure, y_pred, test_set))
            if np.inf in results:
                return np.inf
            results = results / np.max(results)
            len_results = len(results) - 1
            result = np.sum(results) / len_results
            return result
        else:
            results = self.compute_single_score_group(measure, y_pred, test_set)
            if np.inf in results:
                return np.inf
            r1 = np.array([r[0] for r in results])
            r2 = np.array([r[1] for r in results])
            if np.inf in r1 or np.inf in r2:
                return np.inf
            r1 = np.max(r1) - r1
            r2 = np.max(r2) - r2
            r1 = np.sum(r1) / (len(r1) - 1)
            r2 = np.sum(r2) / (len(r2) - 1)
            return (r1 + r2) * 0.5

