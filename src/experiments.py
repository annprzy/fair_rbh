import os
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from neptune.utils import stringify_unsupported
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.classification.logistic_regression import LogisticRegressor
from src.classification.decision_tree import DecisionTree
from src.classification.mlp_classifier import MLPClassifier
from src.classification.naive_bayes import NaiveBayes
from src.datasets.adult_dataset import AdultDataset
from src.datasets.adult_sampled_dataset import AdultSampledDataset
from src.datasets.bank_dataset import BankDataset
from src.datasets.credit_card_dataset import CreditCardDataset
from src.datasets.dataset import Dataset
from src.datasets.german_dataset import GermanDataset
from src.datasets.heart_dataset import HeartDataset
from src.evaluation.logger import init_neptune, log_results
from src.preprocessing.FAWOS import oversamplor as FAWOS
from src.preprocessing.FOS import oversamplor as FOS
from src.preprocessing.FOS_original import oversamplor as FOS_org
from src.preprocessing.HFOS import oversamplor as HFOS
from src.preprocessing.fair_rbo import oversamplor2 as FairRBO


def init_dataset(dataset_name, random_state, data_path='../data'):
    if dataset_name == 'german':
        return GermanDataset(f'{data_path}/german_credit/german.data', binary=True, group_type='',
                             random_state=random_state)
    elif dataset_name == 'bank':
        return BankDataset(f'{data_path}/bank_marketing/bank.csv', binary=True, group_type='',
                           random_state=random_state)
    elif dataset_name == 'adult':
        return AdultSampledDataset(
            f'{data_path}/adult_census/sampled_sex/natural.csv',
            binary=True, group_type='',
            random_state=random_state)
    elif dataset_name == 'credit_card':
        return CreditCardDataset(f'{data_path}/credit_card/credit.data', binary=True, group_type='',
                                 random_state=random_state)
    elif dataset_name == 'heart_disease':
        return HeartDataset(f'{data_path}/heart_disease/processed.cleveland.data', binary=True, group_type='',
                            random_state=random_state)


def run_oversampling(algorithm: str, dataset_name: str, dataset: Dataset, config_path='../configs'):
    if algorithm == 'fos':
        with open(f'{config_path}/preprocessing/fos.yml') as f:
            fos_cfg = yaml.safe_load(f)
        FOS.run(dataset, k=fos_cfg[dataset_name]['k'], oversampling_factor=fos_cfg['oversampling_factor'], distance_type='heom')
    if algorithm == 'hfos':
        with open(f'{config_path}/preprocessing/hfos.yml') as f:
            hfos_cfg = yaml.safe_load(f)
        HFOS.run(dataset, k=hfos_cfg[dataset_name]['k'], distance_type='heom')
    if algorithm == 'fawos':
        with open(f'{config_path}/preprocessing/fawos.yml') as f:
            fawos_cfg = yaml.safe_load(f)
        FAWOS.run(dataset, fawos_cfg[dataset_name]['safe_weight'], fawos_cfg[dataset_name]['borderline_weight'],
                  fawos_cfg[dataset_name]['rare_weight'], oversampling_factor=fawos_cfg['oversampling_factor'], distance_type='heom')
    if algorithm == 'fair_rbo':
        FairRBO.run(dataset, gamma=0.03, approach_number=0)
    return dataset


def fine_tune(algorithm, dataset, classifier):
    small_dataset = deepcopy(dataset)
    training_data = dataset.train
    classes = training_data[dataset.target].to_list()
    group_class = training_data[dataset.sensitive].astype(int).astype(str).agg('-'.join, axis=1).to_list()
    group_class = ['_'.join([g, str(int(c))]) for g, c in zip(group_class, classes)]
    train, test = train_test_split(training_data, test_size=0.3, random_state=42, stratify=group_class)
    small_dataset.data = training_data
    small_dataset.train = train.reset_index(drop=True)
    small_dataset.test = test.reset_index(drop=True)

    best = -np.inf
    best_params = None

    if algorithm == 'fawos':
        weights = [
            {'safe_weight': 0, 'borderline_weight': 0.4, 'rare_weight': 0.6},
            {'safe_weight': 0, 'borderline_weight': 0.5, 'rare_weight': 0.5},
            {'safe_weight': 0, 'borderline_weight': 0.6, 'rare_weight': 0.4},
            {'safe_weight': 0.33, 'borderline_weight': 0.33, 'rare_weight': 0.33},
        ]
        for weight in weights:
            FAWOS.run(small_dataset, weight['safe_weight'], weight['borderline_weight'],
                      weight['rare_weight'], oversampling_factor=1)

            classifier.train(dataset=small_dataset, data='fair', enc_type='cont_ord_cat')
            perf, fairness, _ = classifier.predict_and_evaluate(dataset=small_dataset,
                                                                fairness_type='binary',
                                                                enc_type='cont_ord_cat')
            utility = perf['gmean'] + (1 - np.abs(fairness['average_absolute_odds']))
            if utility > best:
                best_params = weight
                best = utility
        FAWOS.run(dataset, best_params['safe_weight'], best_params['borderline_weight'],
                  best_params['rare_weight'], oversampling_factor=1, distance_type='hvdm')

    elif algorithm == 'fos':
        ks = [3, 5, 7, 11]
        for k in ks:
            FOS.run(small_dataset, k=k, oversampling_factor=1)

            classifier.train(dataset=small_dataset, data='fair', enc_type='cont_ord_cat')
            perf, fairness, _ = classifier.predict_and_evaluate(dataset=small_dataset,
                                                                fairness_type='binary',
                                                                enc_type='cont_ord_cat')
            utility = perf['gmean'] + (1 - np.abs(fairness['average_absolute_odds']))
            if utility > best:
                best_params = k
                best = utility
        FOS.run(dataset, k=best_params, oversampling_factor=1, distance_type='heom')

    elif algorithm == 'hfos':
        ks = [3, 5, 7, 11]
        for k in ks:
            HFOS.run(small_dataset, k=k)

            classifier.train(dataset=small_dataset, data='fair', enc_type='cont_ord_cat')
            perf, fairness, _ = classifier.predict_and_evaluate(dataset=small_dataset,
                                                                fairness_type='binary',
                                                                enc_type='cont_ord_cat')
            utility = perf['gmean'] + (1 - np.abs(fairness['average_absolute_odds']))
            if utility > best:
                best_params = k
                best = utility
        HFOS.run(dataset, k=best_params)

    elif algorithm == 'fair_rbo':
        gammas = [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07]
        for gamma in gammas:
            FairRBO.run(small_dataset, gamma=gamma)

            classifier.train(dataset=small_dataset, data='fair', enc_type='cont_ord_cat')
            perf, fairness, _ = classifier.predict_and_evaluate(dataset=small_dataset,
                                                                fairness_type='binary',
                                                                enc_type='cont_ord_cat')
            utility = perf['gmean'] + (1 - np.abs(fairness['average_absolute_odds']))
            if utility > best:
                best_params = gamma
                best = utility
        FairRBO.run(dataset, gamma=best_params)

    return dataset, best_params


def get_model(model_name: str, config_path='../configs'):
    if model_name == 'logistic_regression':
        return LogisticRegressor(cfg_path=f'{config_path}/classifiers/logistic_regression.yml')
    elif model_name == 'decision_tree':
        return DecisionTree(cfg_path=f'{config_path}/classifiers/decision_tree.yml')
    elif model_name == 'mlp':
        return MLPClassifier(cfg_path=f'{config_path}/classifiers/mlp_classifier.yml')
    elif model_name == 'naive_bayes':
        return NaiveBayes(cfg_path=f'{config_path}/classifiers/naive_bayes.yml')


def experiment(dataset_name: str, algorithm: str, models: list[str], iteration: int, date: str, random_seed: int,
               kfolds: int = None, enc_type: str = 'cont_ord_cat', results_path: str = '../results',
               config_path: str = '../configs', data_path: str = '../data'):
    dataset = init_dataset(dataset_name, random_seed, data_path=data_path)
    if kfolds is not None:
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
        dataset_train = dataset.data
        classes = dataset_train[dataset.target].to_list()
        group_class = dataset_train[dataset.sensitive].astype(int).astype(str).agg('-'.join, axis=1).to_list()
        group_class = ['_'.join([g, str(int(c))]) for g, c in zip(group_class, classes)]
        results = list(kf.split(dataset_train, group_class))[iteration]
        train_set, test_set = results
        dataset.train = dataset_train.iloc[train_set].reset_index(drop=True)
        dataset.test = dataset_train.iloc[test_set].reset_index(drop=True)

    dataset = run_oversampling(algorithm, dataset_name, dataset, config_path=config_path)
    for model_name in models:
        best_params = None
        model = get_model(model_name, config_path=config_path)
        model.load_model()
        #dataset, best_params = fine_tune(algorithm, dataset, model)

        print(
            f'{algorithm}, {dataset_name}, {model_name}, {iteration}, {best_params} \nPrivileged: {dataset.privileged_groups} \n{dataset.train.shape}, {dataset.fair.shape} \nStats train: {dataset.get_stats_data(dataset.train)}Stats fair: {dataset.get_stats_data(dataset.fair)}')

        if not os.path.exists(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/'):
            os.makedirs(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/')
        if not os.path.exists(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}'):
            os.makedirs(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}')

        dataset.train.to_csv(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}/train_{iteration}.csv')
        dataset.fair.to_csv(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}/fair_{iteration}.csv')
        dataset.test.to_csv(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}/test_{iteration}.csv')

        model.train(dataset=dataset, data='train', enc_type=enc_type)
        perf_train_small, fairness_train_small, y_pred_train = model.predict_and_evaluate(dataset=dataset,
                                                                                          fairness_type='binary',
                                                                                          enc_type=enc_type)

        model.train(dataset=dataset, data='fair', enc_type=enc_type)
        perf_fair_small, fairness_fair_small, y_pred_fair = model.predict_and_evaluate(dataset=dataset,
                                                                                       fairness_type='binary',
                                                                                       enc_type=enc_type)

        perf_train_small['data'] = f'train_{enc_type}'
        fairness_train_small['data'] = f'train_{enc_type}'
        perf_fair_small['data'] = f'fair_{enc_type}'
        fairness_fair_small['data'] = f'fair_{enc_type}'
        perf = pd.concat(
            [pd.DataFrame(perf_train_small, index=[iteration]), pd.DataFrame(perf_fair_small, index=[iteration])])
        fairness = pd.concat(
            [pd.DataFrame(fairness_train_small, index=[iteration]),
             pd.DataFrame(fairness_fair_small, index=[iteration])])

        perf.to_csv(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}/performance_{iteration}.csv',
                    index=False)
        fairness.to_csv(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}/fairness_{iteration}.csv',
                        index=False)
        np.save(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}/train_preds_{iteration}.npy',
                y_pred_train)
        np.save(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{date}/fair_preds_{iteration}.npy',
                y_pred_fair)


if __name__ == "__main__":
    datasets = ['heart_disease', 'adult', 'bank', 'german']  #, 'adult', 'bank']
    algorithms = ['hfos', 'fos', 'fawos']  #'fos', 'fawos']
    models = ['logistic_regression', 'decision_tree', 'mlp']
    kfolds = 10
    encoding = 'cont_ord_cat'
    date = '2024-06-19'
    config_path = '../configs'
    results_path = '../results'
    data_path = '../data'
    iterations = [i for i in range(0, kfolds)]
    seeds = [42 for i in iterations]
    all_options = list(product(datasets, algorithms, iterations))
    neptune = False
    with open(f'{config_path}/neptune.yml') as f:
        cfg = yaml.safe_load(f)

    neptune_run = None
    if neptune:
        neptune_run = init_neptune(cfg)
        log_results(neptune_run,
                    {'algorithm': str(algorithms), 'datasets': str(datasets), 'encoding': encoding, 'folds': kfolds},
                    'basic_info')

    Parallel(n_jobs=-1)(delayed(experiment)(d_name, a, models, idx, date, seeds[idx], kfolds=kfolds, enc_type=encoding,
                                            results_path=results_path, config_path=config_path, data_path=data_path) for
                        d_name, a, idx in all_options)

    if neptune:
        for d_name, a, idx in all_options:
            for m_name in models:
                try:
                    perf = pd.read_csv(f'{results_path}/{a}_{d_name}_{m_name}/{date}/performance_{idx}.csv')
                    fairness = pd.read_csv(f'{results_path}/{a}_{d_name}_{m_name}/{date}/fairness_{idx}.csv')
                    log_results(neptune_run, fairness, f'{a}_{d_name}_{m_name}/{date}/fairness_{idx}')
                    log_results(neptune_run, perf, f'{a}_{d_name}_{m_name}/{date}/performance_{idx}')
                except:
                    pass
        neptune_run.stop()
