import pandas as pd

from src.datasets.dataset import Dataset
from src.datasets.transforms import drop_features, attribute_mapper, standard_scaler, one_hot_encoder

CONT = 'continuous'
ORD = 'ordinal'
CAT = 'categorical'


class CreditCardDataset(Dataset):
    def __init__(self, data_path: str = '../../data/credit_card_default/default_of_credit_card_clients.xls',
                 binary=False, group_type='fawos', random_state: int = 42):
        data = pd.read_excel(data_path, header=None, names=['id', 'balance', 'gender', 'education', 'marital', 'age',
                                                            'pay_apr', 'pay_may', 'pay_jun', 'pay_jul', 'pay_aug',
                                                            'pay_sep',
                                                            'bill_amount_apr', 'bill_amount_may', 'bill_amount_jun',
                                                            'bill_amount_jul', 'bill_amount_aug', 'bill_amount_sep',
                                                            'pay_amount_apr', 'pay_amount_may', 'pay_amount_jun',
                                                            'pay_amount_jul', 'pay_amount_aug', 'pay_amount_sep',
                                                            'class'])
        data = data.drop([0, 1])
        data = data.reset_index(drop=True)
        data = drop_features(data, ['id'])
        data, mapping0 = attribute_mapper(data, ['gender', 'education', 'marital'], {
            'gender': {1: 0, 2: 1},
            'education': {4: 0, 3: 1, 1: 2, 2: 3},
            'marital': {1: 1, 2: 0, 3: 0},  # binarize marital status - but maybe better to drop the features?
        })
        mapping = {**mapping0}
        data = data.drop_duplicates(keep='first')
        if binary:
            sensitive_attrs = ['gender']
        else:
            sensitive_attrs = ['gender', 'marital']  # maybe also age?

        target_attr = 'class'
        privileged_class = 1

        feature_types = {
            'balance': CONT,
            'gender': CAT,
            'education': ORD,
            'marital': CAT,
            'age': CONT,
            'pay_apr': ORD,
            'pay_may': ORD,
            'pay_jun': ORD,
            'pay_jul': ORD,
            'pay_aug': ORD,
            'pay_sep': ORD,
            'bill_amount_apr': CONT,
            'bill_amount_may': CONT,
            'bill_amount_jun': CONT,
            'bill_amount_jul': CONT,
            'bill_amount_aug': CONT,
            'bill_amount_sep': CONT,
            'pay_amount_apr': CONT,
            'pay_amount_may': CONT,
            'pay_amount_jun': CONT,
            'pay_amount_jul': CONT,
            'pay_amount_aug': CONT,
            'pay_amount_sep': CONT,
            'class': CAT
        }

        super().__init__(data, sensitive_attrs, target_attr, privileged_class, feature_types,
                         mappings=mapping, group_type=group_type, random_state=random_state)
