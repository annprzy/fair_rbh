import os
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

import src.preprocessing.fair_rbo.oversamplor2 as FairRBO
import src.preprocessing.fair_rbo.hybridsamplor as FairRBH
from src.experiments import get_model, init_dataset
from src.validation_fhf_multi import init_dataset_multi


def check_results(dataset_name, algorithm, distance_type, gamma, approach_number, models, iteration, data_path, config_path, results_path):
    dataset = init_dataset_multi(dataset_name, 42, data_path=data_path)
    date = '2024-06-29'
    enc_type = 'cont_ord_cat'

    params = {'gamma': gamma, 'approach_number': approach_number, 'dist_type': distance_type}


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

    exists_fair = False
    # for model_name in models:
    #     if os.path.exists(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/fair_{iteration}.csv'):
    #         fair_data = pd.read_csv(
    #             f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/fair_{iteration}.csv')
    #         fair_data = fair_data.iloc[:, 1:]
    #         dataset.set_fair(fair_data)
    #         exists_fair = True
    #         break
    if not exists_fair:
        if algorithm == 'fair_rbo':
            FairRBO.run(dataset, distance_type=params['dist_type'], approach_number=params['approach_number'], gamma=params['gamma'])
        else:
            FairRBH.run(dataset, distance_type=params['dist_type'], approach_number=params['approach_number'], gamma=params['gamma'])

    for model_name in models:
        best_params = None
        dataset.train = dataset_train_copy
        model = get_model(model_name, config_path=config_path)
        model.load_model()
        #dataset, best_params = fine_tune(algorithm, dataset, model)

        print(
            f'{algorithm}, {dataset_name}, {model_name}, {iteration}, {params} \nPrivileged: {dataset.privileged_groups} \n{dataset.train.shape}, {dataset.fair.shape} \nStats train: {dataset.get_stats_data(dataset.train)}Stats fair: {dataset.get_stats_data(dataset.fair)}')

        if not os.path.exists(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/'):
            os.makedirs(f'{results_path}/{algorithm}_{dataset_name}_{model_name}/')
        if not os.path.exists(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}'):
            os.makedirs(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}')
        if not os.path.exists(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}'):
            os.makedirs(
                f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}')

        dataset.train.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/train_{iteration}.csv')
        dataset.fair.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/fair_{iteration}.csv')
        dataset.test.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/test_{iteration}.csv')

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
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/performance_{iteration}.csv',
            index=False)
        fairness.to_csv(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/fairness_{iteration}.csv',
            index=False)
        np.save(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/train_preds_{iteration}.npy',
            y_pred_train)
        np.save(
            f'{results_path}/{algorithm}_{dataset_name}_{model_name}/{params["dist_type"]}_{params["gamma"]}_{params["approach_number"]}/{date}/fair_preds_{iteration}.npy',
            y_pred_fair)


if __name__ == '__main__':
    datasets = ['german', 'bank', 'adult']
    gammas = [0.05]
    algorithm = ['fair_rbh']
    distance_metric = {'fair_rbo': ['hvdm', 'heom'], 'fair_rbh': ['hvdm', 'heom']}
    distance_num = [0, 1]
    approach_number = {'fair_rbo': [0, 1, 2, 3, 4], 'fair_rbh': [0, 1, 2, 3, 4]}
    iterations = [0, 1, 2, 3, 4]
    app_ns = [0, 1, 2, 3, 4]
    models = ['logistic_regression', 'decision_tree', 'mlp']
    all_options = list(product(datasets, algorithm, distance_num, gammas, app_ns, iterations))
    config_path = '../configs'
    results_path = '../validation_multi'
    data_path = '../data'

    Parallel(n_jobs=-1)(delayed(check_results)(d_name, alg, distance_metric[alg][dist_type], gamma, approach_number[alg][app_n], models, idx, results_path=results_path,
                                               config_path=config_path, data_path=data_path) for
                        d_name, alg, dist_type, gamma, app_n, idx in all_options)
