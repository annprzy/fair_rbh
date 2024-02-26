import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset
from src.preprocessing.FOS_original.utils import FOS_1, FOS_2


def run(dataset: Dataset):
    protected_attribute = dataset.sensitive[0]

    #######################################################################
    privileged_groups = dataset.privileged_groups
    unprivileged_groups = dataset.unprivileged_groups

    train = dataset.train
    X_train, y_train = dataset.features_and_classes('train')

    n_fav = len(train[train[dataset.target] == dataset.privileged_class])
    n_unfav = len(train[train[dataset.target] == dataset.unprivileged_class])

    n_p_fav = len(train[
                      (train[dataset.target] == dataset.privileged_class) & (
                        train[list(privileged_groups[0].keys())[0]] == list(privileged_groups[0].values())[0])])
    n_p_unfav = len(train[(train[dataset.target] == dataset.unprivileged_class) & (train[
        list(privileged_groups[0].keys())[0]] == list(privileged_groups[0].values())[0])])
    n_up_fav = len(train[(train[dataset.target] == dataset.privileged_class) & (train[
        list(unprivileged_groups[0].keys())[0]] == list(unprivileged_groups[0].values())[0])])
    n_up_unfav = len(train[(train[dataset.target] == dataset.unprivileged_class) & (train[
        list(unprivileged_groups[0].keys())[0]] == list(unprivileged_groups[0].values())[0])])

    prot_idx = X_train.columns.get_loc(protected_attribute)
    prot_values = X_train[protected_attribute].to_numpy()

    pv_max = np.max(prot_values)
    pv_min = np.min(prot_values)
    print('max min ', pv_max, pv_min)

    pv_mid = (pv_max + abs(pv_min)) / 2
    print('mid ', pv_mid)
    pv_mid_pt = pv_max - pv_mid
    print('mid point ', pv_mid_pt)

    #######################################################################
    if n_unfav > n_fav:
        majority = 0
    else:
        majority = 1
    print('majority class ', majority)

    nsamp1 = 0
    cls_trk1 = None
    prot_grp1 = None
    if n_p_fav < n_p_unfav:
        print('first')
        nsamp1 = int(n_p_unfav - n_p_fav)
        prot_grp1 = 1
        if majority == 1:
            cls_trk1 = 1
        else:
            cls_trk1 = 0

    if n_p_unfav < n_p_fav:
        nsamp1 = int(n_p_fav - n_p_unfav)
        prot_grp1 = 1
        if majority == 1:
            cls_trk1 = 0
        else:
            cls_trk1 = 1

    ########################
    nsamp2 = 0
    cls_trk2 = None
    prot_grp2 = None
    if n_up_fav < n_up_unfav:
        nsamp2 = int(n_up_unfav - n_up_fav)
        prot_grp2 = 0
        if majority == 1:
            cls_trk2 = 1
        else:
            cls_trk2 = 0

    if n_up_unfav < n_up_fav:
        nsamp2 = int(n_up_fav - n_up_unfav)
        prot_grp2 = 0
        if majority == 1:
            cls_trk2 = 0
        else:
            cls_trk2 = 1

    if nsamp1 < nsamp2:
        nsamp = nsamp1
        cls_trk = cls_trk1
        prot_grp = prot_grp1
    else:
        nsamp = nsamp2
        cls_trk = cls_trk2
        prot_grp = prot_grp2

    ###################################
    ###################################
    oversampler = FOS_1(random_state=dataset.random_state)

    maj_min = cls_trk

    print('protected group ', prot_grp)
    print('class tracker ', cls_trk)

    print('number to sample ', nsamp)
    print()

    X_samp, y_samp = oversampler.sample(X_train.to_numpy(), y_train.to_numpy(), prot_idx, pv_mid_pt,
                                        prot_grp, maj_min, nsamp, pv_max, pv_min)

    ######################
    print('protected group ', prot_grp)
    print('class tracker ', cls_trk)

    if nsamp1 < nsamp2:
        nsamp = nsamp2
        cls_trk = cls_trk2
        prot_grp = prot_grp2
    else:
        nsamp = nsamp1
        cls_trk = cls_trk1
        prot_grp = prot_grp1

    maj_min = cls_trk

    oversampler = FOS_2(random_state=dataset.random_state)

    X_samp1, y_samp1 = oversampler.sample(X_samp, y_samp, prot_idx, pv_mid_pt,
                                          prot_grp, maj_min, nsamp)

    Xs_train = np.copy(X_samp1)
    ys_train = np.copy(y_samp1)
    train = pd.DataFrame(Xs_train, columns=X_train.columns)
    train[dataset.target] = ys_train
    dataset.set_fair(train)
