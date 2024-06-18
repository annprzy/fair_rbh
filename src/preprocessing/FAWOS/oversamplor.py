import operator
import random
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.classification.classify import Classifier
from src.datasets.dataset import Dataset
from src.preprocessing.FAWOS import taxonomizator
from src.preprocessing.FAWOS.utils import Taxonomy, DatapointsFromClassToOversample, DatapointsToOversample, \
    DatapointAndNeighbours, TaxonomyAndNeighbours


def run(dataset: Dataset, safe_weight: float, borderline_weight: float, rare_weight: float, outlier_weight: float = 0,
        oversampling_factor: float = 1, taxonomies: str | None = None, save_path: str | None = None, distance_type='heom'):
    if taxonomies is None:
        taxonomies = taxonomizator.create_taxonomies_and_neighbours(dataset, distance_type=distance_type)
    datapoints_from_class_to_oversample_list = get_datapoints_from_class_to_oversample_list(dataset, taxonomies,
                                                                                            oversampling_factor)
    train = oversample(dataset, datapoints_from_class_to_oversample_list, safe_weight, borderline_weight,
                       rare_weight, save_path)
    dataset.set_fair(train)


def oversample(dataset: Dataset,
               datapoints_from_class_to_oversample_list: list[DatapointsFromClassToOversample],
               safe_percentage: float,
               borderline_percentage: float,
               rare_percentage: float,
               filename: str | None = None):
    df = dataset.train

    for datapoints_from_class_to_oversample in datapoints_from_class_to_oversample_list:
        datapoints_to_oversample_list = datapoints_from_class_to_oversample.datapoints_to_oversample_list
        random_weights = []
        datapoints_and_neighbours = []
        taxonomies = []
        print("oversampling " + str(datapoints_from_class_to_oversample.classes) + " " +
              str(datapoints_from_class_to_oversample.n_times_to_oversample) + " times")
        for datapoints_to_oversample in datapoints_to_oversample_list:
            taxonomy = datapoints_to_oversample.taxonomy
            if taxonomy == Taxonomy.SAFE:
                weight = safe_percentage
            elif taxonomy == Taxonomy.BORDERLINE:
                weight = borderline_percentage
            elif taxonomy == Taxonomy.RARE:
                weight = rare_percentage
            elif taxonomy == Taxonomy.OUTLIER:
                weight = 0
            else:
                exit("Taxonomy weight not supported " + taxonomy.value)
            random_weights.extend(np.full(len(datapoints_to_oversample.datapoints_and_neighbours), weight))
            datapoints_and_neighbours.extend(datapoints_to_oversample.datapoints_and_neighbours)
        # print(random_weights)

        if datapoints_and_neighbours and random_weights:
            for i in range(datapoints_from_class_to_oversample.n_times_to_oversample):
                # choose random
                if np.sum(random_weights) != 0:
                    proba = np.array(random_weights)
                else:
                    proba = np.ones(len(random_weights))
                random_datapoint_and_neighbour_idx = dataset.random_state.choice(np.arange(0, len(datapoints_and_neighbours)), p=proba / np.sum(proba))
                random_datapoint_and_neighbour = datapoints_and_neighbours[random_datapoint_and_neighbour_idx]
                random_datapoint = random_datapoint_and_neighbour.datapoint
                neighbours = random_datapoint_and_neighbour.neighbours
                random_neighbour = dataset.random_state.choice(np.arange(0, len(neighbours)))
                random_neighbour = neighbours[random_neighbour]

                new_synthetic_datapoint = create_synthetic_sample(dataset.feature_types, random_datapoint,
                                                                  random_neighbour, neighbours, dataset)
                df = pd.concat([df, pd.DataFrame([new_synthetic_datapoint])])

    # save new dataset
    if filename is not None:
        save_dataset(df, filename)
    return df


def save_dataset(dataset: pd.DataFrame, filename: str):
    f = open(filename, "w+")
    f.write(dataset.to_csv(index=False))
    f.close()


def create_synthetic_sample(features: dict, x1, x2, neighbours: list, dataset: Dataset):
    synthetic_example = pd.Series()

    for feature in features.keys():
        x1_value = x1[feature]
        x2_value = x2[feature]

        if features[feature] in ['continuous', 'ordinal']:
            dif = x1_value - x2_value
            gap = dataset.random_state.random()
            synthetic_example_value = x1_value - gap * dif

        # elif features[feature] == 'ordinal':
        #     synthetic_example_value_float = (x1_value + x2_value) / 2
        #     synthetic_example_value = int(synthetic_example_value_float)

        elif features[feature] == 'categorical':
            datapoints = [x1]
            datapoints.extend(neighbours)
            synthetic_example_value = choose_most_common_value_from_all_datapoints(datapoints,
                                                                                   feature)  # TODO does most common??

        else:
            exit("Feature type not valid: " + features[feature])

        synthetic_example[feature] = synthetic_example_value

    return synthetic_example


def choose_most_common_value_from_all_datapoints(datapoints, feature_name):
    counts = {}
    for datapoint in datapoints:
        value = datapoint[feature_name]
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1

    most_common_value = max(counts.items(), key=operator.itemgetter(1))[0]

    return most_common_value


def get_datapoints_from_class_to_oversample_list(dataset: Dataset, taxonomies, oversampling_factor) -> list[
    DatapointsFromClassToOversample]:
    target_class = dataset.target
    positive_class = dataset.privileged_class  # privileged
    negative_class = dataset.unprivileged_class  # unprivileged
    df = dataset.train
    datapoints_from_class_to_oversample_list = []
    unprivileged_classes_combs = dataset.unprivileged_groups
    count_positive_privileged, count_negative_privileged = get_privileged_classes_counts(df, dataset, positive_class,
                                                                                         negative_class)

    for comb in unprivileged_classes_combs:
        df_positive = df.copy(deep=True)
        df_negative = df.copy(deep=True)
        classes = {target_class: [dataset.privileged_class]}

        for class_name, class_values in comb.items():
            class_values = [class_values]
            df_positive = df_positive[
                (df_positive[class_name].isin(class_values)) & (df_positive[target_class] == positive_class)]
            df_negative = df_negative[
                (df_negative[class_name].isin(class_values)) & (df_negative[target_class] == negative_class)]
            classes[class_name] = class_values

        count_positive_unprivileged = len(df_positive)
        count_negative_unprivileged = len(df_negative)

        desired_count_positive_unprivileged = count_negative_unprivileged * count_positive_privileged / count_negative_privileged
        n_times_to_oversample = int(
            (desired_count_positive_unprivileged - count_positive_unprivileged) * oversampling_factor)
        print("Difference in class " + str(classes) + " is " + str(n_times_to_oversample))

        effect = count_negative_unprivileged / count_positive_unprivileged - count_negative_privileged / count_positive_privileged
        print("Effect of class " + str(classes) + " is " + str(effect))

        datapoints_to_oversample_list = []

        for taxonomy in [Taxonomy.OUTLIER, Taxonomy.RARE, Taxonomy.BORDERLINE, Taxonomy.SAFE]:
            datapoints_and_neighbours = get_datapoints_and_neighbours_from_same_classes_and_taxonomy(taxonomies, df,
                                                                                                     classes, taxonomy)
            datapoints_to_oversample = DatapointsToOversample(taxonomy, datapoints_and_neighbours)
            datapoints_to_oversample_list.append(datapoints_to_oversample)

        datapoints_from_class_to_oversample = DatapointsFromClassToOversample(n_times_to_oversample,
                                                                              datapoints_to_oversample_list,
                                                                              classes)
        datapoints_from_class_to_oversample_list.append(datapoints_from_class_to_oversample)

    return datapoints_from_class_to_oversample_list


def get_privileged_classes_counts(df, dataset: Dataset, positive_class, negative_class):
    df_positive = df.copy(deep=True)
    df_negative = df.copy(deep=True)
    target_class = dataset.target
    group = dataset.privileged_groups[0]
    query = [f'`{key}`=={value}' if type(value) is not str else f'`{key}`=="{value}"' for key, value in
             group.items()]
    query = ' and '.join(query)
    df_positive = df_positive.query(query)
    df_negative = df_negative.query(query)
    df_positive = df_positive[df_positive[target_class] == positive_class]
    df_negative = df_negative[df_negative[target_class] == negative_class]

    return len(df_positive), len(df_negative)


def get_datapoints_and_neighbours_from_same_classes_and_taxonomy(taxonomies: list[TaxonomyAndNeighbours],
                                                                 df: pd.DataFrame,
                                                                 classes: dict,
                                                                 taxonomy: Taxonomy) -> list[DatapointAndNeighbours]:
    # taxonomies = dataset.get_taxonomies_and_neighbours()
    indexes_of_datapoints_with_taxonomy = get_indexes_of_datapoints_with_taxonomy(taxonomies, taxonomy)
    df_subset = df.copy(deep=True)
    df_subset = df_subset.loc[indexes_of_datapoints_with_taxonomy]

    for class_name, class_values in dict(classes).items():
        df_subset = df_subset[df_subset[class_name].isin(class_values)]

    datapoints_and_neighbours = []
    datapoint_indexes = df_subset.index.to_list()

    for index in datapoint_indexes:
        neighbours = [df.iloc[index] for index in taxonomies[index].neighbours]
        datapoint = df.iloc[index]
        datapoint_and_neighbours = DatapointAndNeighbours(datapoint, neighbours)
        datapoints_and_neighbours.append(datapoint_and_neighbours)

    return datapoints_and_neighbours


def get_indexes_of_datapoints_with_taxonomy(taxonomies: list[TaxonomyAndNeighbours], taxonomy):
    indexes_of_datapoints_with_taxonomy = []

    for i in range(len(taxonomies)):
        if taxonomies[i].taxonomy == taxonomy:
            indexes_of_datapoints_with_taxonomy.append(i)

    return indexes_of_datapoints_with_taxonomy
