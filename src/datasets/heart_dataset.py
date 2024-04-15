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
                                  'oldpeak', 'slope', 'ca', 'thal', 'class'])
        data.dropna(inplace=True)

        sensitive_attrs = ['sex']

        target_attr = 'class'
        privileged_class = 0
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

        super().__init__(data, sensitive_attrs, target_attr, privileged_class, feature_types,
                         mappings={}, group_type=group_type, random_state=random_state)
