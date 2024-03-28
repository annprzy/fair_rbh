import os
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.classification.logistic_regression import LogisticRegressor
from src.datasets.german_dataset import GermanDataset
from src.preprocessing.FAWOS import oversamplor as FAWOS

if __name__ == '__main__':
    np.random.seed(42)
    dataset = GermanDataset('../data/german_credit/german.data', binary=True, group_type='', random_state=42)
    weights = [
        {'safe': 0, 'borderline': 0.4, 'rare': 0.6},
        {'safe': 0, 'borderline': 0.5, 'rare': 0.5},
        {'safe': 0, 'borderline': 0.6, 'rare': 0.4},
        {'safe': 0.33, 'borderline': 0.33, 'rare': 0.33},
    ]
    dataset_train = deepcopy(dataset.train)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    enc_type = 'cont_ord_cat'
    perf_all = []
    fair_all = []
    for weight in weights:
        i = 0
        perf = []
        fair = []
        for train_index, test_index in kf.split(dataset_train):
            dataset.set_fair([])
            dataset.train = dataset_train.iloc[train_index].reset_index(drop=True)
            dataset.test = dataset_train.iloc[test_index].reset_index(drop=True)
            FAWOS.run(dataset, weight['safe'], weight['borderline'], weight['rare'])

            model = LogisticRegressor(cfg_path='../configs/classifiers/logistic_regression.yml')
            model.load_model()
            model.train(dataset=dataset, data='fair', enc_type=enc_type)
            perf_small, fair_small = model.predict_and_evaluate(dataset=dataset, fairness_type='binary',
                                                                enc_type=enc_type)
            perf_small['weight'] = str(weight)
            fair_small['weight'] = str(weight)
            perf.append(pd.DataFrame(perf_small, index=[i]))
            fair.append(pd.DataFrame(fair_small, index=[i]))
            i += 1

        perf_all.append(pd.concat(perf))
        fair_all.append(pd.concat(fair))
    perf_all = pd.concat(perf_all)
    fair_all = pd.concat(fair_all)

    if not os.path.exists(f'../results/FAWOS_finetune/'):
        os.makedirs(f'../results/FAWOS_finetune/')

    perf_all.to_csv('../results/FAWOS_finetune/performance.csv')
    fair_all.to_csv('../results/FAWOS_finetune/fairness.csv')

