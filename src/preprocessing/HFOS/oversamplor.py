import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset
from src.preprocessing.HFOS.utils import get_clusters, HFOS_SMOTE


def run(dataset: Dataset, k: int = 5):
    X_train, y_train = dataset.features_and_classes("train")
    knn = NearestNeighbors(n_neighbors=k + 1, p=2)
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
                    random_neighbor = h_y.sample(n=1, random_state=dataset.random_state)
                else:
                    random_neighbor = h_g.sample(n=1, random_state=dataset.random_state)
                new_example = hfos_smote.generate_example(random_instance, random_neighbor, dataset)
                ## I dont know actually
                # for f in dataset.sensitive:
                #     new_example[f] = query[f]
                new_examples.append(new_example)
    new_train = pd.concat([dataset.train, *new_examples])
    dataset.set_fair(new_train)

