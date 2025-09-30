import time
import numpy as np

from data_models.cluster import Cluster
from data_models.package import Package
from services.tomtom_client import TomTomClient
from utils.distances_utils import normalize_matrix, get_priority_diversity_matrix


class ClusterManager:
    def __init__(self, packages: list[Package], num_of_clusters, warehouse, clustering_model, tomtom_client: TomTomClient):
        self.packages = packages
        self.num_of_clusters = num_of_clusters
        self.warehouse = warehouse
        self.tomtom_client = tomtom_client
        self.clustering_model = clustering_model
        self.distance_matrix = None
        self.clusters = []


    def build_distance_matrix(self):
        num_packages = len(self.packages)
        self.distance_matrix = np.zeros((num_packages, num_packages))

        chunks = list(chunk_list(self.packages, 50))
        print(len(chunks))
        print([len(c) for c in chunks])

        for i, origins in enumerate(chunks):
            for j, destinations in enumerate(chunks):
                submatrix = self.tomtom_client.get_distance_matrix(origins, destinations)
                for oi, origin in enumerate(origins):
                    for dj, destination in enumerate(destinations):
                        oi_idx = self.packages.index(origin)
                        dj_idx = self.packages.index(destination)
                        self.distance_matrix[oi_idx, dj_idx] = submatrix[oi][dj]
                time.sleep(10)
        return self.distance_matrix


    def build_clusters(self, distance_weight=0.5, priority_weight=0.5):
        priorities = [p.get_priority() for p in self.packages]

        normalized_distances = normalize_matrix(self.distance_matrix)
        normalized_priorities = get_priority_diversity_matrix(priorities)

        similarity = (distance_weight * normalized_distances) + (priority_weight * normalized_priorities)

        labels = self.clustering_model.fit(similarity)

        grouped_packages = [[] for _ in range(self.num_of_clusters)]
        for i, lbl in enumerate(labels):
            grouped_packages[lbl].append(self.packages[i])
            self.packages[i].set_cluster(lbl)

        self.clusters = [Cluster(i, group, self.warehouse) for i, group in enumerate(grouped_packages)]


def chunk_list(lst, size=40):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
