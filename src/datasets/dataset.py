import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from src.datasets.transforms import drop_features


class Dataset:
    def __init__(self, data: pd.DataFrame,
                 sensitive_attrs: list[str],
                 target_attr: str,
                 privileged_class: str | int,
                 feature_types: dict[str, str],
                 privileged_groups: list[dict] = None,
                 unprivileged_groups: list[dict] = None,
                 group_type: str = '',
                 test_set: pd.DataFrame = None,
                 mappings: dict = None,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """Create dataset class from data
        :param data: dataframe containing the dataset
        :param sensitive_attrs: list of sensitive attributes, column names
        :param target_attr: name of target attribute
        :param privileged_class: privileged class
        :param feature_types: types of features in the dataset (continuous, ordinal, categorical)
        :param privileged_groups: list of privileged groups
        :param unprivileged_groups: list of unprivileged groups
        :param mappings: dictionary containing all mappings used during data transformations
        :param test_set: dataframe containing the test set
        :param test_size: the size of the test set (fraction)
        :param random_state: the seed for reproducibility"""
        assert (privileged_groups is not None and unprivileged_groups is not None) or privileged_class is not None
        self.data = data
        self.sensitive = sensitive_attrs
        self.target = target_attr
        self.random_state = np.random.default_rng(random_state)
        self.random_state_init = random_state
        self.mappings = mappings
        self.privileged_class = privileged_class
        self.unprivileged_class = pd.unique(self.data[self.data[self.target] != privileged_class][self.target])[0]
        self.feature_types = feature_types
        self.fair = None
        self.group_type = group_type
        self.test_size = test_size

        self.majority, self.minority = self.compute_min_maj_class(self.data)

        if test_set is not None:
            self.test = test_set
            test_size = 0
        if test_size == 0:
            self.train = data
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size,
                                                     random_state=random_state)

        self.train = self.train.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)

        # for key in standardized_features:
        #     self.test = standardized_features[key](self.train, self.test, [key])
        #     self.train = standardized_features[key](self.train, self.train, [key])

        if privileged_groups is not None:
            self.privileged_groups = privileged_groups
            self.unprivileged_groups = unprivileged_groups
        else:
            if group_type == 'fawos':
                self.privileged_groups, self.unprivileged_groups = self.compute_fawos_groups(self.sensitive, self.train)
            else:
                self.privileged_groups, self.unprivileged_groups = self.compute_groups(self.sensitive, self.train)

    def compute_min_maj_class(self, data: pd.DataFrame) -> tuple[str, str]:
        """compute minority and majority class of dataset
        :param data: dataframe
        :return: tuple (majority class, minority class)"""
        classes_counts = data[self.target].value_counts()
        classes_counts = classes_counts.sort_values(ascending=False)
        return classes_counts.index[0], classes_counts.index[1]

    def compute_groups(self, sensitive: list[str], data: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        """compute privileged and unprivileged groups based on the privileged class and sensitive attributes
        considered
        :param data: dataframe
        :param sensitive: names of sensitive attributes"""
        best_ratio, best_id = -np.inf, None
        not_present = []
        sensitive_values = [list(pd.unique(data[a])) for a in sensitive]
        all_groups = [i for i in itertools.product(*sensitive_values)]
        all_groups = [{a: i for a, i in zip(sensitive, group)} for group in all_groups]
        for i, group in enumerate(all_groups):
            query = [f'`{key}`=={value}' if type(value) is not str else f'`{key}`=="{value}"' for key, value in
                     group.items()]
            query = ' and '.join(query)
            classes = data.query(query)
            if len(classes) != 0:
                priv = len(classes[classes[self.target] == self.privileged_class])
                unpriv = len(classes[classes[self.target] != self.privileged_class])
                if priv != 0 and unpriv != 0:
                    ratio = priv / unpriv
                    if ratio > best_ratio:
                        best_id = i
                        best_ratio = ratio
                else:
                    not_present.append(i)
            else:
                not_present.append(i)
        privileged_groups = [all_groups[best_id]]
        unprivileged_groups = [g for i, g in enumerate(all_groups) if i != best_id and i not in not_present]
        return privileged_groups, unprivileged_groups

    def compute_privileged_unprivileged_values(self, sensitive: list[str], data: pd.DataFrame) -> tuple[list, list]:
        """compute privileged and unprivileged values for all sensitive attributes
        :param data: dataframe
        :param sensitive: list of names of sensitive attributes
        :return: privileged and unprivileged values for each attribute"""
        p_values = []
        up_values = []
        for attr in sensitive:
            p, up = self.compute_groups([attr], data)
            p = [i[attr] for i in p]
            up = [i[attr] for i in up]
            p_values.append(p)
            up_values.append(up)
        return p_values, up_values

    def compute_fawos_groups(self, sensitive: list[str], data: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        """compute privileged and unprivileged groups as in FAWOS
        :param data: dataframe
        :param sensitive: sensitive attributes
        :return: privileged and unprivileged groups"""
        priv, _ = self.compute_privileged_unprivileged_values(sensitive, data)
        priv = [i for i in itertools.product(*priv)]
        sensitive_values = [list(pd.unique(data[a])) for a in sensitive]
        all_groups = [i for i in itertools.product(*sensitive_values)]
        privileged_groups = [{a: i for a, i in zip(sensitive, group)} for group in priv]
        unprivileged_groups = [{a: i for a, i in zip(sensitive, group)} for group in all_groups if group not in priv]
        return privileged_groups, unprivileged_groups

    def set_fair(self, df: pd.DataFrame):
        self.fair = df

    def features_and_classes(self, data_type: str, encoding: bool = False, enc_type: str = None, shuffle: bool = False):
        if data_type == "train":
            if shuffle:
                train = self.train.sample(frac=1)
            else:
                train = self.train
            X = train.loc[:, self.train.columns != self.target]
            if encoding:
                X = self.perform_encoding(enc_type, X, X)
            y = train[self.target]
        elif data_type == "test":
            X = self.test.loc[:, self.test.columns != self.target]
            if encoding:
                X = self.perform_encoding(enc_type, self.train.loc[:, self.train.columns != self.target], X)
            y = self.test[self.target]
        elif data_type == "fair":
            X = self.fair.loc[:, self.fair.columns != self.target]
            if encoding:
                X = self.perform_encoding(enc_type, self.train.loc[:, self.train.columns != self.target], X)
            y = self.fair[self.target]
        else:
            raise ValueError(f"data type {data_type} not supported")
        return X, y

    def perform_encoding(self, calc_type: str, df_to_fit: pd.DataFrame, df_to_transform: pd.DataFrame):
        X = deepcopy(df_to_transform)
        if calc_type == 'cont_ord_cat' or calc_type == 'cont_ord':
            features_cont_ord = [f for f, v in self.feature_types.items() if
                                 (v == 'continuous' or v == 'ordinal') and f != self.target]
        elif calc_type == 'cont':
            features_cont_ord = [f for f, v in self.feature_types.items() if
                                 (v == 'continuous') and f != self.target]
        else:
            raise ValueError(f"Wrong type for performing encoding {calc_type}")
        features_cat = [f for f, v in self.feature_types.items() if
                        v == 'categorical' and len(self.data[f].unique()) > 2 and f != self.target]
        scaler = MinMaxScaler()
        scaler.fit(df_to_fit[features_cont_ord])
        X.loc[:, features_cont_ord] = scaler.transform(df_to_transform[features_cont_ord])
        if calc_type == 'cont_ord_cat':
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(self.data[features_cat])
            features_names = encoder.get_feature_names_out(features_cat)
            new_data = encoder.transform(df_to_transform[features_cat]).toarray()
            X.loc[:, features_names] = new_data
            X = drop_features(X, features_cat)
        return X

    def get_config(self):
        cfg = {
            'sensitive_attributes': self.sensitive,
            'group_type': self.group_type,
            'privileged_groups': self.privileged_groups,
            'unprivileged_groups': self.unprivileged_groups,
            'minority': self.minority,
            'majority': self.majority,
            'privileged_class': self.privileged_class,
            'unprivileged_class': self.unprivileged_class,
            'seed': self.random_state_init,
            'test_size': self.test_size,
            'mapping': self.mappings
        }
        return cfg

    def get_stats_data(self, data: pd.DataFrame) -> str:
        num_majority = len(data[data[self.target] == self.majority])
        num_minority = len(data) - num_majority
        result = f''
        result += f'Num minority: {num_minority}\n'
        result += f'Num majority: {num_majority}\n'
        for group in [*self.privileged_groups, *self.unprivileged_groups]:
            examples = len(query_dataset(group, data))
            examples_min = len(query_dataset({**group, self.target: self.minority}, data))
            examples_maj = len(query_dataset({**group, self.target: self.majority}, data))
            result += f'Group: {group}, Len: {examples}, Minority: {examples_min}, Majority: {examples_maj}\n'
        return result


def query_dataset(query: dict, df: pd.DataFrame) -> pd.DataFrame:
    query = [f'`{key}`=={value}' if type(value) is not str else f'`{key}`=="{value}"' for key, value in
             query.items()]
    query = ' and '.join(query)
    return df.query(query)
