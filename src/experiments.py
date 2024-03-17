import os

import numpy as np
import pandas as pd
import yaml

from src.classification.logistic_regression import LogisticRegressor
from src.datasets.adult_dataset import AdultDataset
from src.datasets.bank_dataset import BankDataset
from src.datasets.credit_card_dataset import CreditCardDataset
from src.datasets.german_dataset import GermanDataset
from src.evaluation.logger import init_neptune, log_results
from src.preprocessing.FAWOS import oversamplor as FAWOS
from src.preprocessing.FOS import oversamplor as FOS
from src.preprocessing.FOS_original import oversamplor as FOS_org
from src.preprocessing.HFOS import oversamplor as HFOS

if __name__ == "__main__":
    np.random.seed(42)
    neptune = True
    with open('../configs/neptune.yml') as f:
        cfg = yaml.safe_load(f)

    for algorithm in ['hfos']:
        german_dataset = GermanDataset('../data/german_credit/german.data', binary=True, group_type='')
        adult_dataset = AdultDataset('../data/adult_census/adult.test', binary=True, group_type='')
        bank_dataset = BankDataset('../data/bank_marketing/bank.csv', binary=True, group_type='')
        # credit_dataset = CreditCardDataset('../data/credit_card_default/default_of_credit_card_clients.xls',
        #                                    binary=True, group_type='')
        datasets = [('german', german_dataset), ('bank-small', bank_dataset), ('adult-small', adult_dataset), ]  # ('adult', adult_dataset),
        for t in datasets:
            data_name, dataset = t

            if neptune:
                neptune_run = init_neptune(cfg)
                log_results(neptune_run, {'algorithm': algorithm, 'data': data_name, 'encoding': 'no_onehot_only_cont'}, 'basic_info')

            print(algorithm, data_name)
            print(f'before: {dataset.privileged_groups}')

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
                FAWOS.run(dataset, fawos_cfg['safe_weight'], fawos_cfg['borderline_weight'], fawos_cfg['rare_weight'], oversampling_factor=fawos_cfg['oversampling_factor'])

            print(f'after: {dataset.privileged_groups}')
            print(dataset.train.shape, dataset.fair.shape)
            print('Stats train:')
            dataset.get_stats_data(dataset.train)
            print('Stats fair:')
            dataset.get_stats_data(dataset.fair)

            perf_train, fairness_train, perf_fair, fairness_fair = [], [], [], []

            for i, enc_type in enumerate(['cont_ord', 'cont', 'cont_ord_cat']):

                model = LogisticRegressor(cfg_path='../configs/classifiers/logistic_regression.yml')
                model.load_model()
                model.train(dataset=dataset, data='train', enc_type=enc_type)
                perf_train_small, fairness_train_small = model.predict_and_evaluate(dataset=dataset, fairness_type='binary', enc_type=enc_type)

                model = LogisticRegressor(cfg_path='../configs/classifiers/logistic_regression.yml')
                model.load_model()
                model.train(dataset=dataset, data='fair', enc_type=enc_type)
                perf_fair_small, fairness_fair_small = model.predict_and_evaluate(dataset=dataset, fairness_type='binary', enc_type=enc_type)

                perf_train_small['data'] = f'train_{enc_type}'
                fairness_train_small['data'] = f'train_{enc_type}'
                perf_fair_small['data'] = f'fair_{enc_type}'
                fairness_fair_small['data'] = f'fair_{enc_type}'

                perf_fair.append(pd.DataFrame(perf_fair_small, index=[i]))
                perf_train.append(pd.DataFrame(perf_train_small, index=[i]))
                fairness_train.append(pd.DataFrame(fairness_train_small, index=[i]))
                fairness_fair.append(pd.DataFrame(fairness_fair_small, index=[i]))

            perf_train = pd.concat(perf_train)
            fairness_train = pd.concat(fairness_train)
            perf_fair = pd.concat(perf_fair)
            fairness_fair = pd.concat(fairness_fair)

            if neptune:
                log_results(neptune_run, perf_train, 'performance_raw')
                log_results(neptune_run, fairness_train, 'fairness_raw')

                log_results(neptune_run, perf_fair, 'performance_fair')
                log_results(neptune_run, fairness_fair, 'fairness_fair')

            perf = pd.concat([perf_train, perf_fair])
            fairness = pd.concat([fairness_train, fairness_fair])
            if not os.path.exists(f'../results/{algorithm}_{data_name}_lr/'):
                os.makedirs(f'../results/{algorithm}_{data_name}_lr/')
            if algorithm == 'hfos':
                perf.to_csv(f'../results/{algorithm}_{data_name}_lr/performance_only_cont_int.csv', index=False)
                fairness.to_csv(f'../results/{algorithm}_{data_name}_lr/fairness_only_cont_int.csv', index=False)
            else:
                perf.to_csv(f'../results/{algorithm}_{data_name}_lr/performance.csv', index=False)
                fairness.to_csv(f'../results/{algorithm}_{data_name}_lr/fairness.csv', index=False)
            if neptune:
                neptune_run.stop()
