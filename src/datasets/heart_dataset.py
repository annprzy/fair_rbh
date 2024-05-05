import pandas as pd

from src.datasets.dataset import Dataset
from src.datasets.transforms import attribute_mapper

CONT = 'continuous'
ORD = 'ordinal'
CAT = 'categorical'


class HeartDataset(Dataset):
    def __init__(self, data_path: str = '../../data/heart_disease/processed.cleveland.data', binary=False,
                 group_type='fawos', random_state: int = 42):
        data = pd.read_csv(data_path, header=None,
                           names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restcg', 'thalach', 'exang',
                                  'oldpeak', 'slope', 'ca', 'thal', 'class'], na_values=['?'])
        data.dropna(inplace=True)
        
        data, mapping0 = attribute_mapper(data, ['class'], {'class': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}})

        sensitive_attrs = ['sex']

        target_attr = 'class'
        privileged_class = 0
        
        #data = data.drop(columns=['cp', 'fbs', 'restcg', 'exang', 'slope', 'thal'])
        
        data = data.drop_duplicates(keep='first')
        data = data.drop_duplicates(subset=[c for c in data.columns if c != target_attr], keep='first')

        feature_types = {
            'age': CONT,
            'sex': CAT,
            'cp': CAT,
            'trestbps': CONT,
            'chol': CONT,
            'fbs': CAT,
            'restcg': CAT,
            'thalach': CONT,
            'exang': CAT,
            'oldpeak': CONT,
            'slope': CAT,
            'ca': ORD,
            'thal': CAT,
            'class': CAT
        }
        
        data.reset_index(drop=True, inplace=True)

        super().__init__(data, sensitive_attrs, target_attr, privileged_class, feature_types,
                         mappings=mapping0, group_type=group_type, random_state=random_state)
