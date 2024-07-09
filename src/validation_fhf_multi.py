import os
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

import src.preprocessing.HFOS.oversamplor as HFOS
import src.preprocessing.FOS.oversamplor as FOS
import src.preprocessing.FAWOS.oversamplor as FAWOS
from src.datasets.adult_sampled_dataset import AdultSampledDataset
from src.datasets.bank_dataset import BankDataset
from src.datasets.credit_card_dataset import CreditCardDataset
from src.datasets.german_dataset import GermanDataset
from src.datasets.heart_dataset import HeartDataset
from src.experiments import get_model


def init_dataset_multi(dataset_name, random_state, data_path='../data'):
    if dataset_name == 'german':
        return GermanDataset(f'{data_path}/german_credit/german.data', binary=False, group_type='',
                             random_state=random_state)
    elif dataset_name == 'bank':
        return BankDataset(f'{data_path}/bank_marketing/bank_multi.csv', binary=False, group_type='',
                           random_state=random_state)
    elif dataset_name == 'adult':
        return AdultSampledDataset(
            f'{data_path}/adult_census/sampled_sex/natural.csv',
            binary=False, group_type='',
            random_state=random_state)


def check_results(dataset_name, distance_type, algorithm, models, iteration, tested_value, data_path, config_path,
                  results_path, perform_fair=True):
    dataset = init_dataset_multi(dataset_name, 42, data_path=data_path)
    date = '2024-06-12'
    enc_type = 'cont_ord_cat'

    params = {'dist_type': distance_type, 'tested_values': tested_value}

    kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    dataset_train = dataset.data
    classes = dataset_train[dataset.target].to_list()
    group_class = dataset_train[dataset.sensitive].astype(int).astype(str).agg('-'.join, axis=1).to_list()
    group_class = ['_'.join([g, str(int(c))]) for g, c in zip(group_class, classes)]
    results = list(kf.split(dataset_train, group_class))[0]
    train_set, _ = results
    dataset_train_sample = dataset_train.iloc[train_set].reset_index(drop=True)
    kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    group_class = [g for i, g in enumerate(group_class) if i in train_set]
    results = list(kf.split(dataset_train_sample, group_class))[iteration]
    train_set, test_set = results
    dataset.train = dataset_train_sample.iloc[train_set].reset_index(drop=True)
    dataset.test = dataset_train_sample.iloc[test_set].reset_index(drop=True)
    dataset_train_copy = dataset_train_sample.iloc[train_set].reset_index(drop=True)

    if perform_fair:
        if algorithm == 'hfos':
            HFOS.run(dataset, k=params['tested_values'], distance_type=params['dist_type'])
        else:
            FAWOS.run(dataset, distance_type=params['dist_type'], safe_weight=params['tested_values']['safe_weight'],
                      borderline_weight=params['tested_values']['borderline_weight'],
                      rare_weight=params['tested_values']['rare_weight'])

    for model_name in models:
        best_params = None
        dataset.train = dataset_train_copy
        model = get_model(model_name, config_path=config_path)
        model.load_model()
        #dataset, best_params = fine_tune(algorithm, dataset, model)
        if not perform_fair:
            fair_data = pd.read_csv(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/fair_{iteration}.csv')
            fair_data = fair_data.iloc[:, 1:]
            dataset.set_fair(fair_data)

        print(
            f'{algorithm}, {dataset_name}, {model_name}, {iteration}, {params} \nPrivileged: {dataset.privileged_groups} \n{dataset.train.shape}, {dataset.fair.shape} \nStats train: {dataset.get_stats_data(dataset.train)}Stats fair: {dataset.get_stats_data(dataset.fair)}')

        if not os.path.exists(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/'):
            os.makedirs(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/')
        if not os.path.exists(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}'):
            os.makedirs(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}')
        if not os.path.exists(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}'):
            os.makedirs(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}')
        if not os.path.exists(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}'):
            os.makedirs(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}')

        dataset.train.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/train_{iteration}.csv')
        dataset.fair.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/fair_{iteration}.csv')
        dataset.test.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/test_{iteration}.csv')

        model.train(dataset=dataset, data='train', enc_type=enc_type)
        perf_train_small, fairness_train_small, y_pred_train = model.predict_and_evaluate(dataset=dataset,
                                                                                          fairness_type='multi',
                                                                                          enc_type=enc_type)

        model.train(dataset=dataset, data='fair', enc_type=enc_type)
        perf_fair_small, fairness_fair_small, y_pred_fair = model.predict_and_evaluate(dataset=dataset,
                                                                                       fairness_type='multi',
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

        perf.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/performance_{iteration}.csv',
            index=False)
        fairness.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/fairness_{iteration}.csv',
            index=False)
        np.save(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/train_preds_{iteration}.npy',
            y_pred_train)
        np.save(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}/{params["tested_values"]}/{date}/fair_preds_{iteration}.npy',
            y_pred_fair)


if __name__ == '__main__':
    datasets = ['adult']
    distance_metric = {'fawos': ['heom', 'hvdm'], 'hfos': ['hvdm', 'heom']}
    algorithm_name = ['fawos', 'hfos']
    # tested_values = [
    #         {'safe_weight': 0, 'borderline_weight': 0.4, 'rare_weight': 0.6},
    #         {'safe_weight': 0, 'borderline_weight': 0.5, 'rare_weight': 0.5},
    #         {'safe_weight': 0, 'borderline_weight': 0.6, 'rare_weight': 0.4},
    #         {'safe_weight': 0.33, 'borderline_weight': 0.33, 'rare_weight': 0.33},
    # ]
    # #tested_values = [3, 5, 7, 11]
    # tested_values_num = [0, 1, 2, 3]
    tested_values = {
        'fawos': [{'safe_weight': 0, 'borderline_weight': 0.4, 'rare_weight': 0.6}],
        'hfos': [5]
    }
    #tested_values = [{'safe_weight': 0, 'borderline_weight': 0.4, 'rare_weight': 0.6}]
    #tested_values = [5]
    tested_values_num = [0]
    distance_metric_num = [0, 1]
    iterations = [0, 1, 2, 3, 4]
    models = ['logistic_regression', 'decision_tree', 'mlp']
    all_options = list(product(datasets, distance_metric_num, algorithm_name, tested_values_num, iterations))
    config_path = '../configs'
    results_path = '../validation_multi'
    data_path = '../data'
    perform_fair = True

    Parallel(n_jobs=-1)(
        delayed(check_results)(d_name, distance_metric[app_n][dist_type], app_n, models, idx, tested_values[app_n][t],
                               results_path=results_path,
                               config_path=config_path, data_path=data_path, perform_fair=perform_fair)
        for
        d_name, dist_type, app_n, t, idx in all_options)
