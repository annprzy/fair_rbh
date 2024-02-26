from enum import Enum

import numpy as np


class Taxonomy(Enum):
    SAFE = "Safe"
    BORDERLINE = "Bordeline"
    RARE = "Rare"
    OUTLIER = "Outlier"


class TaxonomyAndNeighbours:

    def __init__(self, taxonomy: Taxonomy, neighbours: list[int]):
        self.taxonomy = taxonomy
        self.neighbours = neighbours

    @staticmethod
    def read_taxonomies_and_neighbours(filename: str) -> list:
        f = open(filename, "r")
        text = f.read()

        taxonomies_and_neighbours = []
        for line in text.split("\n"):
            line = line.split(",")
            taxonomy = line[0]
            if line[1:] != ['']:
                neighbours = [int(l) for l in line[1:]]
            else:
                neighbours = []  # outlier
            taxonomy_and_neighbours = TaxonomyAndNeighbours(Taxonomy(taxonomy), neighbours)
            taxonomies_and_neighbours.append(taxonomy_and_neighbours)

        f.close()

        return taxonomies_and_neighbours

    @staticmethod
    def save_taxonomies_and_neighbours(filename: str, taxonomies_and_neighbours: list) -> None:
        f = open(filename, "w+")

        for i in range(len(taxonomies_and_neighbours)):
            f.write(taxonomies_and_neighbours[i].taxonomy.value)
            f.write(",")
            f.write(",".join([str(neighbour) for neighbour in taxonomies_and_neighbours[i].neighbours]))
            if i != len(taxonomies_and_neighbours) - 1:
                f.write("\n")

        f.close()


class DatapointAndNeighbours:

    def __init__(self, datapoint: np.array, neighbours: list[np.array]) -> None:
        self.datapoint = datapoint
        self.neighbours = neighbours


class DatapointsToOversample:

    def __init__(self, taxonomy: Taxonomy, datapoints_and_neighbours: list[DatapointAndNeighbours]) -> None:
        self.taxonomy = taxonomy
        self.datapoints_and_neighbours = datapoints_and_neighbours


class DatapointsFromClassToOversample:

    def __init__(self, n_times_to_oversample: int, datapoints_to_oversample_list: list[DatapointsToOversample],
                 classes: dict) -> None:
        self.n_times_to_oversample = n_times_to_oversample
        self.datapoints_to_oversample_list = datapoints_to_oversample_list
        self.classes = classes
