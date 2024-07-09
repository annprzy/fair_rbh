import pandas as pd

from src.datasets.dataset import Dataset
from src.datasets.transforms import attribute_mapper, ordinal_encoder, standard_scaler, drop_features, one_hot_encoder

CONT = 'continuous'
ORD = 'ordinal'
CAT = 'categorical'


class AdultSampledDataset(Dataset):
    def __init__(self, data_path: str = '../../data/adult_census/adult.data', binary=False, group_type='fawos',
                 random_state: int = 42, attr_binary: str | None = None):
        data = pd.read_csv(data_path, header=0, na_values=['?'])
        data.dropna(inplace=True)
        data = data.drop_duplicates(keep='first')
        if binary and attr_binary is None:
            sensitive_attrs = ['sex']
        elif binary and attr_binary is not None:
            sensitive_attrs = [attr_binary]
        else:
            sensitive_attrs = ['race', 'sex']

        target_attr = 'class'
        privileged_class = 1

        # data = data.drop(columns=['workclass', 'marital', 'occupation', 'relationship', 'sex', 'native_country'])

        data = data.drop_duplicates(keep='first')
        data = data.drop_duplicates(subset=[c for c in data.columns if c != target_attr], keep='first')

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

        data = data.astype(float)
        data.reset_index(drop=True, inplace=True)

        super().__init__(data, sensitive_attrs, target_attr, privileged_class, feature_types,
                         mappings={}, group_type=group_type, random_state=random_state)