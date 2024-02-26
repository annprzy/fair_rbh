import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, MinMaxScaler, StandardScaler


def attribute_mapper(data: pd.DataFrame, attrs: list[str], mapping: dict) -> tuple[pd.DataFrame, dict]:
    """
    :param data: dataframe
    :param attrs: names of attributes to perform the mapping on
    :param mapping: dictionary containing information about the mapping
    :return: transformed data, mapping
    """
    for attr in attrs:
        data[attr] = data[attr].map(mapping[attr])
    return data, mapping


def one_hot_encoder(data: pd.DataFrame, attrs: list[str]) -> tuple[pd.DataFrame, dict]:
    """encode the given attribute using one hot encoding
    :param data: dataframe
    :param attrs: list of names of columns to perform the encoding on
    :return: transformed data, empty dictionary
    """
    enc = OneHotEncoder()
    enc.fit(data[attrs])
    features_names = enc.get_feature_names_out(attrs)
    data.loc[:, features_names] = enc.transform(data[attrs])
    return data, {}


def ordinal_encoder(data: pd.DataFrame, attrs: list[str]) -> tuple[pd.DataFrame, dict]:
    """encode the given attribute using ordinal encoding
    :param data: dataframe
    :param attrs: list of names of columns to perform the encoding on
    :return: transformed data, dict with mappings
    """
    enc = OrdinalEncoder(dtype=np.int32)
    enc.fit(data[attrs])
    data.loc[:, attrs] = enc.transform(data[attrs])
    return data, {attr: enc.categories_[i] for i, attr in enumerate(attrs)}


def target_binarizer(data: pd.DataFrame, attrs: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """binarize the target attribute
    :param data: dataframe
    :param attrs: list with the name of the target attribute
    :return: transformed data, the label for each class"""
    lb = LabelBinarizer()
    lb.fit(data[attrs])
    data[attrs] = lb.transform(data[attrs])
    return data, lb.classes_


def minmax_scaler(fit_data: pd.DataFrame, transform_data: pd.DataFrame, attrs: list[str]) -> pd.DataFrame:
    """min-max scaling in range 0 to 1
    :param fit_data: dataframe to fit
    :param transform_data: dataframe to transform
    :param attrs: list of names of columns to perform the scaling on
    :return: transformed data
    """
    scaler = MinMaxScaler()
    scaler.fit(fit_data[attrs])
    transform_data[attrs] = scaler.transform(transform_data[attrs])
    return transform_data


def standard_scaler(fit_data: pd.DataFrame, transform_data: pd.DataFrame, attrs: list[str]) -> pd.DataFrame:
    """standardize features by removing the mean and scaling to unit variance
    :param fit_data: dataframe to fit
    :param transform_data: dataframe to transform
    :param attrs: list of names of columns to perform the scaling on
    """
    scaler = StandardScaler()
    scaler.fit(fit_data[attrs])
    transform_data[attrs] = scaler.transform(transform_data[attrs])
    return transform_data


def discretizer(data: pd.DataFrame, attrs: list[str], mapping: dict) -> tuple[pd.DataFrame, dict]:
    """discretize based on given mapping
    :param data: dataframe
    :param attrs: names of attributes to perform the mapping on
    :param mapping: dictionary containing the discrete values as the keys and list of two values (range, left inclusive, right exclusive) for each
    discrete value, for instance {0: [0, 20], 1: [20, 40], 2:[40, 60]}
    :return: transformed dataframe, mapping
    """
    for attr in attrs:
        attribute = data[attr]
        new_attribute = []
        for a in attribute:
            for key, value in mapping[attr].items():
                if value[0] <= a < value[1]:
                    new_attribute.append(key)
                    break
        data[attr] = new_attribute
    return data, {attr: mapping[attr] for attr in attrs}


def drop_features(data: pd.DataFrame, f_to_drop: list[str]) -> pd.DataFrame:
    """drop features from dataframe
    :param data: dataframe
    :param f_to_drop: names of features to drop
    :return: transformed dataframe"""
    data = data.drop(columns=f_to_drop, inplace=False)
    return data


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    """remove rows containing nan values
    :param data: dataframe
    :return: transformed data"""
    data = data.dropna(inplace=False)
    return data
