import numpy as np
import pandas as pd
import yaml

from src.classification.logistic_regression import LogisticRegressor
from src.datasets.bank_dataset import BankDataset
from src.datasets.german_dataset import GermanDataset
from src.evaluation.logger import init_neptune, log_results
from src.preprocessing.FAWOS import oversamplor as FAWOS
from src.preprocessing.FOS import oversamplor as FOS
from src.preprocessing.FOS_original import oversamplor as FOS_org
from src.preprocessing.HFOS import oversamplor as HFOS

if __name__ == "__main__":
    np.random.seed(42)
    with open('../configs/neptune.yml') as f:
        cfg = yaml.safe_load(f)

    neptune_run = init_neptune(cfg)

    dataset = BankDataset('../data/bank_marketing/bank.csv', binary=False, group_type='')

    log_results(neptune_run, {'algorithm': 'hfos', 'data': 'bank.csv', 'calc_type_metric': 'sonoda'}, 'basic_info')

    HFOS.run(dataset, k=5)
    print(dataset.train.shape, dataset.fair.shape)
    # log_results(neptune_run, dataset.get_config(), 'dataset_config')

    model = LogisticRegressor(cfg_path='../configs/classifiers/logistic_regression.yml')
    model.load_model()
    model.train(dataset=dataset, data='train')
    perf_train, fairness_train = model.predict_and_evaluate(dataset=dataset, fairness_type='multi', calc_type='sonoda')

    model = LogisticRegressor(cfg_path='../configs/classifiers/logistic_regression.yml')
    model.load_model()
    model.train(dataset=dataset, data='fair')
    perf_fair, fairness_fair = model.predict_and_evaluate(dataset=dataset, fairness_type='multi', calc_type='sonoda')

    perf_train['data'] = 'train'
    fairness_train['data'] = 'train'
    perf_fair['data'] = 'fair'
    fairness_fair['data'] = 'fair'

    log_results(neptune_run, perf_train, 'performance_raw')
    log_results(neptune_run, fairness_train, 'fairness_raw')

    log_results(neptune_run, perf_fair, 'performance_fair')
    log_results(neptune_run, fairness_fair, 'fairness_fair')

    perf = pd.DataFrame([perf_train, perf_fair])
    fairness = pd.DataFrame([fairness_train, fairness_fair])
    perf.to_csv('../results/performance_knn_try.csv', index=False)
    fairness.to_csv('../results/fairness_knn_try.csv', index=False)
