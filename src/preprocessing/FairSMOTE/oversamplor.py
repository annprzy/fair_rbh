import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset
from src.preprocessing.FairSMOTE.utils import generate_samples
from src.preprocessing.HFOS.utils import get_clusters


def run(dataset: Dataset, cr: float = 0.8, f: float = 0.8):
    new_data = []
    subgroups = get_clusters(dataset)
    max_cluster_len = -np.inf
    for data in subgroups:
        _, c, _, _ = data
        if len(c) > max_cluster_len:
            max_cluster_len = len(c)

    for data in subgroups:
        query, df, _, _ = data
        to_oversample = max_cluster_len - len(df)
        new_df = generate_samples(dataset, df, to_oversample, cr=cr, f=f)
        new_data.append(new_df)

    new_data = pd.concat(new_data)
    dataset.set_fair(new_data)
