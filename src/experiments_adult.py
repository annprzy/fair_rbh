import os
from itertools import product

import pandas as pd
import yaml
from joblib import Parallel, delayed
from pandas.core.common import random_state
from sklearn.model_selection import KFold

from src.datasets.adult_dataset import AdultDataset
from src.evaluation.logger import init_neptune, log_results
from src.experiments import run_oversampling, get_model


def experiment_adult(dataset_name: str, algorithm: str, models: list[str], iteration: int, date: str, random_seed: int,
                     kfolds: int = None, enc_type: str = 'cont_ord_cat', results_path: str = '../results',
                     config_path: str = '../configs', data_path: str = '../data'):
    dataset = AdultDataset(f'{data_path}/{dataset_name}.csv', binary=True, group_type='', random_state=random_seed)
    if kfolds is not None:
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        dataset_train = dataset.data
        results = list(kf.split(dataset_train))[iteration]
        train_set, test_set = results
        dataset.train = dataset_train.iloc[train_set].reset_index(drop=True)
        dataset.test = dataset_train.iloc[test_set].reset_index(drop=True)

    dataset = run_oversampling(algorithm, dataset, config_path=config_path)

    for model_name in models:
        print(
            f'{algorithm}, {dataset_name}, {model_name}, {iteration} \nPrivileged: {dataset.privileged_groups} \n{dataset.train.shape}, {dataset.fair.shape} \nStats train: {dataset.get_stats_data(dataset.train)}Stats fair: {dataset.get_stats_data(dataset.fair)}')

        if not os.path.exists(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/'):
            os.makedirs(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/')
        if not os.path.exists(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/{date}'):
            os.makedirs(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/{date}')

        dataset.train.to_csv(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/{date}/train_{iteration}.csv')
        dataset.fair.to_csv(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/{date}/fair_{iteration}.csv')

        model = get_model(model_name, config_path=config_path)
        model.load_model()

        model.train(dataset=dataset, data='train', enc_type=enc_type)
        perf_train_small, fairness_train_small = model.predict_and_evaluate(dataset=dataset,
                                                                            fairness_type='binary',
                                                                            enc_type=enc_type)

        model.train(dataset=dataset, data='fair', enc_type=enc_type)
        perf_fair_small, fairness_fair_small = model.predict_and_evaluate(dataset=dataset,
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

        perf.to_csv(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/{date}/performance_{iteration}.csv',
                    index=False)
        fairness.to_csv(f'{results_path}/{algorithm}_adult_{dataset_name}_{model_name}/{date}/fairness_{iteration}.csv',
                        index=False)


if __name__ == "__main__":
    algorithms = ['new_thing', 'hfos', 'fawos', 'fos', 'fair_smote',
                  'fawos_hybrid']  #'hfos', 'fos']  #, 'fos', 'fawos']
    models = ['logistic_regression', 'decision_tree', 'mlp', 'naive_bayes']
    kfolds = 10
    encoding = 'cont_ord_cat'
    date = '2024-05-20'
    config_path = '../configs'
    results_path = '../results'
    data_path = '../data'
    folder_path = 'sampled_sex'
    dataset_files = [f for f in os.listdir(f'{data_path}/adult_census/{folder_path}')]
    iterations = [i for i in range(0, kfolds)]
    seeds = [42 + i for i in iterations]
    all_options = list(product(dataset_files, algorithms, iterations))
    neptune = False
    with open(f'{config_path}/neptune.yml') as f:
        cfg = yaml.safe_load(f)

    neptune_run = None
    if neptune:
        neptune_run = init_neptune(cfg)
        log_results(neptune_run,
                    {'algorithm': str(algorithms), 'datasets': str(dataset_files), 'encoding': encoding,
                     'folds': kfolds},
                    'basic_info')

    Parallel(n_jobs=-1)(
        delayed(experiment_adult)(d_name, a, models, idx, date, seeds[idx], kfolds=kfolds, enc_type=encoding,
                                  results_path=results_path, config_path=config_path,
                                  data_path=f'{data_path}/adult_census/{folder_path}') for
        d_name, a, idx in all_options)

    if neptune:
        for d_name, a, idx in all_options:
            for m_name in models:
                try:
                    perf = pd.read_csv(f'{results_path}/{a}_adult_{d_name}_{m_name}/{date}/performance_{idx}.csv')
                    fairness = pd.read_csv(f'{results_path}/{a}_adult_{d_name}_{m_name}/{date}/fairness_{idx}.csv')
                    log_results(neptune_run, fairness, f'{a}_adult_{d_name}_{m_name}/{date}/fairness_{idx}')
                    log_results(neptune_run, perf, f'{a}_adult_{d_name}_{m_name}/{date}/performance_{idx}')
                except:
                    pass
        neptune_run.stop()
