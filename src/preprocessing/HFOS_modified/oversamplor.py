import numpy as np
import pandas as pd
from distython import HEOM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters
from src.preprocessing.HFOS_modified.utils import HFOS_SMOTE


def run(dataset: Dataset, k: int = 5):
    X_train, y_train = dataset.features_and_classes("train")
    cat_ord_features = [f for f, t in dataset.feature_types.items() if
                        (t == 'ordinal' or t == 'categorical') and f != dataset.target]
    cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
    metric = HEOM(X_train, cat_ord_features, nan_equivalents=[np.nan])
    knn = NearestNeighbors(n_neighbors=k + 1, metric=metric.heom, n_jobs=-1)
    #knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    knn.fit(X_train)

    clusters = get_clusters(dataset)
    max_cluster_len = -np.inf
    for data in clusters:
        _, c, _, _ = data
        if len(c) > max_cluster_len:
            max_cluster_len = len(c)
    new_examples = []
    hfos_smote = HFOS_SMOTE(k, knn)
    for data in clusters:
        query, cluster, h_y, h_g = data
        if len(cluster) > 0:
            to_generate = max_cluster_len - len(cluster)
            random_instances = cluster.sample(n=to_generate, replace=True, random_state=dataset.random_state)
            p_y_g = len(h_y) / (len(h_y) + len(h_g))
            for idx, random_instance in random_instances.iterrows():
                random_instance = random_instance.to_frame().T
                which_cluster = dataset.random_state.choice([1, 0], size=1, p=[p_y_g, 1 - p_y_g])
                if which_cluster == 1:
                    random_cluster = h_y
                else:
                    random_cluster = h_g
                new_example = hfos_smote.generate_example(random_instance, random_cluster, dataset)
                ## I dont know actually
                for f in dataset.sensitive:
                    new_example[f] = query[f]
                new_examples.append(new_example)
    new_train = pd.concat([dataset.train, *new_examples])
    dataset.set_fair(new_train)
