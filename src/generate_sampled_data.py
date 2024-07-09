import numpy as np

from src.datasets.adult_dataset import AdultDataset
from src.datasets.dataset import Dataset, query_dataset
from src.datasets.sample_dataset import get_number_samples, sample_data


def binary_proportions(dataset: Dataset):
    # giving exact proportions
    natural = {}
    for g in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
        for c in [dataset.majority, dataset.minority]:
            query = {**g, dataset.target: c}
            len_subgroup = len(query_dataset(query, dataset.data))
            key = '_'.join([str(i) for i in query.values()])
            natural[key] = len_subgroup

    group_imbalance = {}
    # considering group imbalances
    group_imbalance['strongly_imbalanced_privileged'] = {
        '_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 9,
        '_'.join([str(k) for k in dataset.unprivileged_groups[0].values()]): 1}
    group_imbalance['mildly_imbalanced_privileged'] = {
        '_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 7,
        '_'.join(
            [str(k) for k in dataset.unprivileged_groups[0].values()]): 3}
    group_imbalance['balanced_g'] = {'_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 1,
                                     '_'.join([str(k) for k in dataset.unprivileged_groups[0].values()]): 1}
    # group_imbalance['mildly_imbalanced_unprivileged'] = {
    #     '_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 4,
    #     '_'.join([str(k) for k in dataset.unprivileged_groups[0].values()]): 6}
    # group_imbalance['strongly_imbalanced_unprivileged'] = {
    #     '_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 1,
    #     '_'.join([str(k) for k in dataset.unprivileged_groups[0].values()]): 9}

    class_imbalance = {}
    # considering class imbalances
    class_imbalance['strongly_imbalanced_maj'] = {str(dataset.majority): 9, str(dataset.minority): 1}
    class_imbalance['mildly_imbalanced_maj'] = {str(dataset.majority): 6, str(dataset.minority): 4}
    class_imbalance['balanced_c'] = {str(dataset.majority): 1, str(dataset.minority): 1}
    class_imbalance['mildly_imbalanced_min'] = {str(dataset.majority): 4, str(dataset.minority): 6}
    class_imbalance['strongly_imbalanced_min'] = {str(dataset.majority): 1, str(dataset.minority): 9}

    # considering group and class imbalances
    all_imbalances = {}
    for g in group_imbalance:
        for c in class_imbalance:
            all_imbalances['_'.join([g, c])] = [group_imbalance[g], class_imbalance[c]]

    return natural, group_imbalance, class_imbalance, all_imbalances


def sth_sth(dataset):
    group_imbalance = {'strongly_imbalanced_g': [1, 9], 'mildly_imbalanced_g': [3, 7], 'balanced_g': [1, 1]}
    groups_sort = {}
    for g in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
        len_group = len(query_dataset(g, dataset.data))
        groups_sort['_'.join([str(i) for i in g.values()])] = len_group
    sorted_groups = np.argsort(np.array(list(groups_sort.values())))
    for g in group_imbalance:
        values = sorted(group_imbalance[g])
        order = {}
        for v, i in zip(values, sorted_groups):
            key = list(groups_sort.keys())[i]
            order[key] = v
        group_imbalance[g] = order

    class_imbalance = {}
    class_imbalance['strongly_imbalanced_c'] = {str(dataset.majority): 9, str(dataset.minority): 1}
    class_imbalance['mildly_imbalanced_c'] = {str(dataset.majority): 6, str(dataset.minority): 4}
    class_imbalance['balanced_c'] = {str(dataset.majority): 1, str(dataset.minority): 1}

    # for imb in class_imbalance:
    #     natural = {}
    #     imb_sum = class_imbalance[imb][str(dataset.majority)] + class_imbalance[imb][str(dataset.minority)]
    #     for g in [*dataset.privileged_groups, *dataset.unprivileged_groups]:
    #         len_group = len(query_dataset(g, dataset.data))
    #         len_minority = len_group * class_imbalance[imb][str(dataset.minority)] / imb_sum
    #         len_majority = len_group * class_imbalance[imb][str(dataset.majority)] / imb_sum
    #         key_minority = '_'.join([str(i) for i in {**g, dataset.target: dataset.minority}.values()])
    #         key_majority = '_'.join([str(i) for i in {**g, dataset.target: dataset.majority}.values()])
    #         natural[key_minority] = len_minority
    #         natural[key_majority] = len_majority
    #     class_imbalance[imb] = natural
    all_imbalances = {}
    for g in group_imbalance:
        for c in class_imbalance:
            all_imbalances['_'.join([g, c])] = [group_imbalance[g], class_imbalance[c]]
    return all_imbalances


if __name__ == '__main__':
    random_state = 42
    data_path = '../data'
    number_examples = 2000
    binary_adult_sex = AdultDataset(f'{data_path}/adult_census/adult.data', binary=True, group_type='',
                                    random_state=random_state)
    natural, group_imbalance, class_imbalance, all_imbalances = binary_proportions(binary_adult_sex)

    number_samples_natural = get_number_samples(number_examples, binary_adult_sex, group_class_imbalance=natural)
    #sample_data(binary_adult_sex, number_samples_natural, save_path=f'{data_path}/adult_census/natural.csv')

    # for g, d in group_imbalance.items():
    #     number_samples = get_number_samples(number_examples, binary_adult_sex, group_imbalance=d)
    #     sample_data(binary_adult_sex, number_samples, save_path=f'{data_path}/adult_census/sampled_sex/new/{g}.csv')
    #
    all_imbalances = sth_sth(binary_adult_sex)
    print(all_imbalances)
    # for c, d in class_imbalance.items():
    #     number_samples = get_number_samples(number_examples, binary_adult_sex, group_class_imbalance=d)
    #     print(number_samples)
    #     sample_data(binary_adult_sex, number_samples, save_path=f'{data_path}/adult_census/sampled_sex/new/{c}.csv')

    for k, d in all_imbalances.items():
        g, c = d
        number_samples = get_number_samples(number_examples, binary_adult_sex, class_imbalance=c, group_imbalance=g)
        sample_data(binary_adult_sex, number_samples, save_path=f'{data_path}/adult_census/sampled_sex/new/{k}.csv')
