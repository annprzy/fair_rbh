import os
from itertools import product

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from neptune.utils import stringify_unsupported
from sklearn.model_selection import KFold

from src.classification.logistic_regression import LogisticRegressor
from src.datasets.adult_dataset import AdultDataset
from src.datasets.bank_dataset import BankDataset
from src.datasets.credit_card_dataset import CreditCardDataset
from src.datasets.dataset import Dataset
from src.datasets.german_dataset import GermanDataset
from src.evaluation.logger import init_neptune, log_results
from src.preprocessing.FAWOS import oversamplor as FAWOS
from src.preprocessing.FOS import oversamplor as FOS
from src.preprocessing.FOS_original import oversamplor as FOS_org
from src.preprocessing.HFOS import oversamplor as HFOS


def init_dataset(dataset_name, random_state):
    if dataset_name == 'german':
        return GermanDataset('../data/german_credit/german.data', binary=True, group_type='', random_state=random_state)
    elif dataset_name == 'bank':
        return BankDataset('../data/bank_marketing/bank-full.csv', binary=True, group_type='',
                           random_state=random_state)
    elif dataset_name == 'adult':
        return AdultDataset('../data/adult_census/adult.data', binary=True, group_type='', random_state=random_state)


def run_oversampling(algorithm: str, dataset: Dataset):
    if algorithm == 'fos':
        with open('../configs/preprocessing/fos.yml') as f:
            fos_cfg = yaml.safe_load(f)
        FOS.run(dataset, k=fos_cfg['k'], oversampling_factor=fos_cfg['oversampling_factor'])
    if algorithm == 'hfos':
        with open('../configs/preprocessing/hfos.yml') as f:
            hfos_cfg = yaml.safe_load(f)
        HFOS.run(dataset, k=hfos_cfg['k'])
    if algorithm == 'fawos':
        with open('../configs/preprocessing/fawos.yml') as f:
            fawos_cfg = yaml.safe_load(f)
        FAWOS.run(dataset, fawos_cfg['safe_weight'], fawos_cfg['borderline_weight'],
                  fawos_cfg['rare_weight'], oversampling_factor=fawos_cfg['oversampling_factor'])
    return dataset


def get_model(model_name: str):
    if model_name == 'logistic_regression':
        return LogisticRegressor(cfg_path='../configs/classifiers/logistic_regression.yml')


def experiment(dataset_name: str, algorithm: str, model_name: str, iteration: int, date: str, random_seed: int,
               kfolds: int = None, enc_type: str = 'cont_ord_cat'):
    dataset = init_dataset(dataset_name, random_seed)
    if kfolds is not None:
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        dataset_train = dataset.train
        results = list(kf.split(dataset_train))[iteration]
        train_set, test_set = results
        dataset.train = dataset_train.iloc[train_set].reset_index(drop=True)
        dataset.test = dataset_train.iloc[test_set].reset_index(drop=True)

    dataset = run_oversampling(algorithm, dataset)
    print(f'{algorithm}, {dataset_name}, {model_name}, {iteration} \nPrivileged: {dataset.privileged_groups} \n{dataset.train.shape}, {dataset.fair.shape} \nStats train: {dataset.get_stats_data(dataset.train)}Stats fair: {dataset.get_stats_data(dataset.fair)}')

    model = get_model(model_name)
    model.load_model()
    model.train(dataset=dataset, data='train', enc_type=enc_type)
    perf_train_small, fairness_train_small = model.predict_and_evaluate(dataset=dataset,
                                                                        fairness_type='binary',
                                                                        enc_type=enc_type)

    model = get_model(model_name)
    model.load_model()
    model.train(dataset=dataset, data='fair', enc_type=enc_type)
    perf_fair_small, fairness_fair_small = model.predict_and_evaluate(dataset=dataset,
                                                                      fairness_type='binary',
                                                                      enc_type=enc_type)

    perf_train_small['data'] = f'train_{enc_type}'
    fairness_train_small['data'] = f'train_{enc_type}'
    perf_fair_small['data'] = f'fair_{enc_type}'
    fairness_fair_small['data'] = f'fair_{enc_type}'
    perf = pd.concat([pd.DataFrame(perf_train_small, index=[iteration]), pd.DataFrame(perf_fair_small, index=[iteration])])
    fairness = pd.concat([pd.DataFrame(fairness_train_small, index=[iteration]), pd.DataFrame(fairness_fair_small, index=[iteration])])

    if not os.path.exists(f'../results/{algorithm}_{dataset_name}_{model_name}/'):
        os.makedirs(f'../results/{algorithm}_{dataset_name}_{model_name}/')
    if not os.path.exists(f'../results/{algorithm}_{dataset_name}_{model_name}/{date}'):
        os.makedirs(f'../results/{algorithm}_{dataset_name}_{model_name}/{date}')
    perf.to_csv(f'../results/{algorithm}_{dataset_name}_{model_name}/performance_{iteration}.csv',
                index=False)
    fairness.to_csv(f'../results/{algorithm}_{dataset_name}_{model_name}/fairness_{iteration}.csv',
                    index=False)


if __name__ == "__main__":
    datasets = ['german'] #, 'adult', 'bank']
    algorithms = ['hfos'] #, 'fos', 'fawos']
    models = ['logistic_regression']
    kfolds = 2
    encoding = 'cont_ord_cat'
    date = '2023-03-28'
    iterations = [i for i in range(0, kfolds)]
    seeds = [42+i for i in iterations]
    all_options = list(product(datasets, algorithms, models, iterations, seeds))
    neptune = True
    with open('../configs/neptune.yml') as f:
        cfg = yaml.safe_load(f)

    neptune_run = None
    if neptune:
        neptune_run = init_neptune(cfg)
        log_results(neptune_run, {'algorithm': str(algorithms), 'datasets': str(datasets), 'encoding': encoding, 'folds': kfolds}, 'basic_info')

    Parallel(n_jobs=-1)(delayed(experiment)(d_name, a, m_name, idx, date, rnd_seed, kfolds=kfolds, enc_type=encoding) for d_name, a, m_name, idx, rnd_seed in all_options)

    if neptune:
        for d_name, a, m_name, idx, rnd_seed in all_options:
            perf = pd.read_csv(f'../results/{a}_{d_name}_{m_name}/performance_{idx}.csv')
            fairness = pd.read_csv(f'../results/{a}_{d_name}_{m_name}/fairness_{idx}.csv')
            log_results(neptune_run, fairness, f'{a}_{d_name}_{m_name}/fairness_{idx}')
            log_results(neptune_run, perf, f'{a}_{d_name}_{m_name}/performance_{idx}')
        neptune_run.stop()
