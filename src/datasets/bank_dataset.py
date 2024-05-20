import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset
from src.datasets.transforms import discretizer, attribute_mapper, ordinal_encoder, drop_features, standard_scaler, \
    one_hot_encoder

CONT = 'continuous'
ORD = 'ordinal'
CAT = 'categorical'


class BankDataset(Dataset):
    def __init__(self, data_path: str = '../../data/bank_marketing/bank-full.csv', binary=False, group_type='fawos', random_state: int = 42):
        data = pd.read_csv(data_path, sep=';')
        data.rename(columns={'y': 'class'}, inplace=True)
        data, mapping0 = discretizer(data, ['age'], {'age': {0: [0, 25], 1: [25, np.inf]}})
        data, mapping1 = attribute_mapper(data,
                                          ['marital', 'education', 'default', 'housing', 'loan', 'month', 'class'],
                                          {'marital': {'married': 1, 'divorced': 0,
                                                       'single': 0},
                                           'education': {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3},
                                           'default': {'yes': 1, 'no': 0},
                                           'housing': {'yes': 1, 'no': 0},
                                           'loan': {'yes': 1, 'no': 0},
                                           'month': {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
                                                     'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11},
                                           'class': {'yes': 1, 'no': 0},
                                           })
        data, mapping3 = ordinal_encoder(data, ['job', 'contact', 'poutcome'])
        mapping = {**mapping0, **mapping1, **mapping3}
        if binary:
            sensitive_attrs = ['age']
        else:
            sensitive_attrs = ['age', 'marital']

        target_attr = 'class'
        privileged_class = 1

        data = drop_features(data, ['day', 'month'])
        data = data.drop_duplicates(keep='first')
        data = data.drop_duplicates(subset=[c for c in data.columns if c != target_attr], keep='first')
        feature_types = {
            'age': CAT,
            'job': CAT,
            'marital': CAT,
            'education': ORD,
            'default': CAT,
            'balance': CONT,
            'housing': CAT,
            'loan': CAT,
            'contact': CAT,
            # 'day': CONT,
            # 'month': CAT,
            'duration': CONT,
            'campaign': CONT,
            'pdays': CONT,
            'previous': CONT,
            'poutcome': CAT,
            'class': CAT
        }

        data = data.astype(float)
        data.reset_index(drop=True, inplace=True)

        super().__init__(data, sensitive_attrs, target_attr, privileged_class, feature_types,
                         mappings=mapping, group_type=group_type, random_state=random_state)
