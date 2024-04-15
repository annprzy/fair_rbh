import numpy as np
import pandas as pd
from distython import HEOM
from sklearn.neighbors import NearestNeighbors

from src.datasets.dataset import Dataset
from src.preprocessing.FAWOS.utils import TaxonomyAndNeighbours, Taxonomy


def create_taxonomies_and_neighbours(dataset: Dataset,
                                     taxonomies_filename: str | None = None):
    X_train, y_train = dataset.features_and_classes("train")
    # distances = heomDist(dataset, X_train)

    cat_ord_features = [f for f, t in dataset.feature_types.items() if
                        (t == 'ordinal' or t == 'categorical') and f != dataset.target]
    cat_ord_features = [X_train.columns.get_loc(c) for c in cat_ord_features]
    metric = HEOM(X_train, cat_ord_features, nan_equivalents=[np.nan])
    knn = NearestNeighbors(n_neighbors=6, metric=metric.heom, n_jobs=-1)
    knn.fit(X_train)

    taxonomies_and_neighbours = determine_taxonomies_and_neighbours(knn, dataset, X_train, y_train)
    if taxonomies_filename is not None:
        TaxonomyAndNeighbours.save_taxonomies_and_neighbours(taxonomies_filename, taxonomies_and_neighbours)
    else:
        return taxonomies_and_neighbours


def determine_taxonomies_and_neighbours(knn,
                                        dataset: Dataset,
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series) -> list[TaxonomyAndNeighbours]:
    taxonomies_and_neighbours = []
    neighbours_list = calculate_neighbours(knn, dataset, X_train, y_train)

    for i in range(len(neighbours_list)):
        count = len(neighbours_list[i])
        taxonomy = ""

        if count == 0:
            taxonomy = Taxonomy.OUTLIER

        elif count == 1:
            solo_neighbour = neighbours_list[i][0]
            neighbours_of_solo_neighbour = neighbours_list[solo_neighbour]

            if len(neighbours_of_solo_neighbour) == 0 or (
                    len(neighbours_of_solo_neighbour) == 1 and neighbours_of_solo_neighbour[0] == i):
                taxonomy = Taxonomy.RARE

            else:
                taxonomy = Taxonomy.BORDERLINE

        elif count in [2, 3]:
            taxonomy = Taxonomy.BORDERLINE

        elif count in [4, 5]:
            taxonomy = Taxonomy.SAFE

        taxonomy_and_neighbours = TaxonomyAndNeighbours(taxonomy, neighbours_list[i])
        taxonomies_and_neighbours.append(taxonomy_and_neighbours)

    return taxonomies_and_neighbours


def calculate_neighbours(knn, dataset: Dataset, X_train: pd.DataFrame, y_train: pd.Series):
    neighbours_list = []

    distances, nearest_neighbors = knn.kneighbors(X_train)

    for idx, distance, nns in zip(range(len(X_train)), distances, nearest_neighbors):
        datapoint = X_train.iloc[idx, :]
        target_datapoint = y_train.iloc[idx]
        distance = distance.flatten()
        nns = nns.flatten()
        nns = np.array([X_train.index[n] for n, d in zip(nns, distance) if d > 0])
        assert len(nns) == 5, (distances, datapoint)

        neighbours = []

        for neighbour in nns:
            neighbour_datapoint = X_train.iloc[neighbour, :]
            target_neighbour_datapoint = y_train.iloc[neighbour]

            points_belong_to_same_sensitive_classes = True
            for sensitive_class in dataset.sensitive:
                if datapoint[sensitive_class] != neighbour_datapoint[sensitive_class]:
                    points_belong_to_same_sensitive_classes = False
                    break

            if points_belong_to_same_sensitive_classes and target_datapoint == target_neighbour_datapoint:
                neighbours.append(neighbour)

        neighbours_list.append(neighbours)

    return neighbours_list


def heomDist(dataset: Dataset, X_train: pd.DataFrame):
    val_max_col = []
    val_min_col = []

    for col in X_train.columns:
        max = X_train[col].max()
        val_max_col.append(max)
        min = X_train[col].min()
        val_min_col.append(min)

    N = X_train.shape[0]
    distances = np.full((N, N,), np.inf)

    for i in range(N):
        if i % 10 == 0:
            print("Calculating distances of point:", i)
        for j in range(i + 1, N):
            distance = heom(dataset, X_train, val_max_col, val_min_col, i, j)
            distances[i][j] = distance
            distances[j][i] = distance

    return distances


def heom(dataset: Dataset, data, val_max_col, val_min_col, m, n):
    dist_sum = 0
    i = 0
    for feature, feature_type in dataset.feature_types.items():
        if feature != dataset.target:
            mm = data[feature].iloc[m]
            nn = data[feature].iloc[n]

            dist_temp = 0
            if mm == '' or nn == '':
                dist_temp = 1

            # 'binary', 'categorical', 'ordinal'
            elif feature_type == 'categorical' or feature_type == 'ordinal':
                if mm == nn:
                    dist_temp = 0
                else:
                    dist_temp = 1

            # 'interval', 'continuous'
            elif feature_type == 'continuous':
                if val_max_col[i] - val_min_col[i] == 0:
                    dist_temp = 0
                else:
                    dist_temp = (float(mm) - float(nn)) / (val_max_col[i] - val_min_col[i])

            else:
                exit("Taxonomize Exception - " + mm + " or " + nn + "values not recognized")

            dist_sum += dist_temp ** 2
            i += 1

    dist_heom = dist_sum ** 0.5

    return dist_heom
