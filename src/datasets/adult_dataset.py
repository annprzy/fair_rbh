import pandas as pd

from src.datasets.dataset import Dataset
from src.datasets.transforms import attribute_mapper, ordinal_encoder, standard_scaler, drop_features, one_hot_encoder

CONT = 'continuous'
ORD = 'ordinal'
CAT = 'categorical'


class AdultDataset(Dataset):
    def __init__(self, data_path: str = '../../data/adult_census/adult.data', binary=False, group_type='fawos', random_state: int = 42):
        data = pd.read_csv(data_path, header=None,
                           names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation',
                                  'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                                  'native_country', 'class'], na_values=['?'])
        data.dropna(inplace=True)
        data, mapping0 = attribute_mapper(data, ['education', 'race', 'sex', 'class'], {
            'education': {' Preschool': 0, ' 1st-4th': 1, ' 5th-6th': 2, ' 7th-8th': 3, ' 9th': 4, ' 10th': 5, ' 11th': 6, ' 12th': 7, ' HS-grad': 8, ' Some-college': 9, ' Assoc-voc': 10, ' Assoc-acdm': 11, ' Bachelors': 12, ' Masters': 13, ' Doctorate': 14, ' Prof-school': 15},
            'race': {' White': 0, ' Asian-Pac-Islander': 1, ' Amer-Indian-Eskimo': 1, ' Black': 1, ' Other': 1},
            'sex': {' Female': 0, ' Male': 1},
            'class': {' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1}
        })
        data = drop_features(data, ['fnlwgt'])
        data, mapping1 = ordinal_encoder(data, ['workclass', 'marital', 'occupation', 'relationship', 'native_country'])
        mapping = {**mapping0, **mapping1}
        data = data.drop_duplicates(keep='first')
        if binary:
            sensitive_attrs = ['race']
        else:
            sensitive_attrs = ['race', 'sex']

        target_attr = 'class'
        privileged_class = 1

        feature_types = {
            'age': CONT,
            'workclass': CAT,
            'education': ORD,
            'education_num': CONT,
            'marital': CAT,
            'occupation': CAT,
            'relationship': CAT,
            'race': CAT,
            'sex': CAT,
            'capital_gain': CONT,
            'capital_loss': CONT,
            'hours_per_week': CONT,
            'native_country': CAT,
            'class': CAT
        }

        super().__init__(data, sensitive_attrs, target_attr, privileged_class, feature_types,
                         mappings=mapping, group_type=group_type, random_state=random_state)
