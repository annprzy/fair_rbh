import pandas as pd

from src.datasets.dataset import Dataset
from src.datasets.transforms import *

CONT = 'continuous'
ORD = 'ordinal'
CAT = 'categorical'


class GermanDataset(Dataset):
    def __init__(self, data_path: str = '../../data/german_credit/german.data', binary=False, group_type='fawos'):
        data = pd.read_csv(data_path,
                           header=None,
                           sep=' ',
                           names=['account_status', 'months', 'credit_history', 'credit_purpose', 'credit_amount',
                                  'savings', 'employment', 'installment_rate', 'personal_status_sex', 'debtors',
                                  'residence', 'property', 'age', 'installment_plans', 'housing', 'number_credits',
                                  'job', 'liables', 'phone', 'foreign_worker', 'class'])
        data['personal_status'] = data['personal_status_sex'][:]
        data['sex'] = data['personal_status_sex'][:]
        data, mapping0 = attribute_mapper(data, ['personal_status', 'sex', 'savings', 'account_status', 'class'], {
            'personal_status': {'A91': 'not_single', 'A92': 'not_single', 'A93': 'single', 'A94': 'not_single',
                                'A95': 'single'},
            'sex': {'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male', 'A95': 'female'},
            'savings': {'A65': 0, 'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4},
            'account_status': {'A11': 0, 'A14': 1, 'A12': 2, 'A13': 3},
            'class': {1: 1, 2: 0},  # 1 is the positive class, 2 is the negative class
        })
        data = drop_features(data, ['personal_status_sex', 'personal_status'])
        data, mapping1 = ordinal_encoder(data,
                                         [  # 'account_status',
                                             'credit_history', 'credit_purpose',
                                             # 'savings',
                                             'employment',
                                             # 'personal_status',
                                             'sex', 'debtors',
                                             'property', 'housing', 'phone', 'foreign_worker', 'installment_plans',
                                             'job'])
        data, mapping2 = discretizer(data, ['age'], {'age': {1: [0, 25], 0: [25, np.inf]}})
        mapping = {**mapping0, **mapping1, **mapping2}
        if binary:
            sensitive_attrs = ['sex']
        else:
            sensitive_attrs = ['age', 'sex']

        target_attr = 'class'
        privileged_class = 1

        feature_types = {
            'account_status': ORD,
            'months': CONT,
            'credit_history': CAT,
            'credit_purpose': CAT,
            'credit_amount': CONT,
            'savings': ORD,
            'employment': ORD,
            'installment_rate': CONT,
            # 'personal_status': CAT,
            'sex': CAT,
            'debtors': CAT,
            'residence': CONT,
            'property': CAT,
            'age': CAT,
            'installment_plans': CAT,
            'housing': CAT,
            'number_credits': CONT,
            'job': ORD,
            'liables': CONT,
            'phone': CAT,
            'foreign_worker': CAT,
            'class': CAT
        }

        standardized_features = {
            'months': standard_scaler,
            'credit_amount': standard_scaler
        }

        super().__init__(data, sensitive_attrs, target_attr, privileged_class, feature_types, standardized_features,
                         mappings=mapping, group_type=group_type)
