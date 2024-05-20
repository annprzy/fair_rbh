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
    group_imbalance['mildly_imbalanced_privileged'] = {'_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 6,
                                                       '_'.join(
                                                           [str(k) for k in dataset.unprivileged_groups[0].values()]): 4}
    group_imbalance['balanced_g'] = {'_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 1,
                                     '_'.join([str(k) for k in dataset.unprivileged_groups[0].values()]): 1}
    group_imbalance['mildly_imbalanced_unprivileged'] = {
        '_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 4,
        '_'.join([str(k) for k in dataset.unprivileged_groups[0].values()]): 6}
    group_imbalance['strongly_imbalanced_unprivileged'] = {
        '_'.join([str(k) for k in dataset.privileged_groups[0].values()]): 1,
        '_'.join([str(k) for k in dataset.unprivileged_groups[0].values()]): 9}

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


if __name__ == '__main__':
    random_state = 42
    data_path = '../data'
    number_examples = 5000
    binary_adult_sex = AdultDataset(f'{data_path}/adult_census/adult.data', binary=True, group_type='',
                                    random_state=random_state, attr_binary='sex')
    print(binary_adult_sex.get_stats_data(binary_adult_sex.data))
    natural, group_imbalance, class_imbalance, all_imbalances = binary_proportions(binary_adult_sex)

    number_samples_natural = get_number_samples(number_examples, binary_adult_sex, group_class_imbalance=natural)
    sample_data(binary_adult_sex, number_samples_natural, save_path=f'{data_path}/adult_census/sampled_sex/natural.csv')

    for g, d in group_imbalance.items():
        number_samples = get_number_samples(number_examples, binary_adult_sex, group_imbalance=d)
        sample_data(binary_adult_sex, number_samples, save_path=f'{data_path}/adult_census/sampled_sex/{g}.csv')

    for c, d in class_imbalance.items():
        number_samples = get_number_samples(number_examples, binary_adult_sex, class_imbalance=d)
        sample_data(binary_adult_sex, number_samples, save_path=f'{data_path}/adult_census/sampled_sex/{c}.csv')

    for k, d in all_imbalances.items():
        g, c = d
        number_samples = get_number_samples(number_examples, binary_adult_sex, class_imbalance=c, group_imbalance=g)
        sample_data(binary_adult_sex, number_samples, save_path=f'{data_path}/adult_census/sampled_sex/{k}.csv')
