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
        self.distance_matrix = None
        self.clustering_model = clustering_model
        self.clusters = []


    def build_distance_matrix(self):
        job_id = self.tomtom_client.submit_matrix_routing_request(self.packages)
        matrix_routing_response = self.tomtom_client.poll_matrix_routing_result(job_id)
        self.distance_matrix = np.array(self.tomtom_client.response_to_result_matrix(matrix_routing_response))


    def build_clusters(self):
        priorities = [p.get_priority() for p in self.packages]

        distance_weight, priority_weight = 0.8, 0.2

        normalized_distances = normalize_matrix(self.distance_matrix)
        normalized_priorities = get_priority_diversity_matrix(priorities)

        similarity = (distance_weight * normalized_distances) + (priority_weight * normalized_priorities)

        labels = self.clustering_model.fit(similarity)

        grouped_packages = [[] for _ in range(self.num_of_clusters)]
        for i, lbl in enumerate(labels):
            grouped_packages[lbl].append(self.packages[i])
            self.packages[i].set_cluster(lbl)

        self.clusters = [Cluster(i, group, self.warehouse) for i, group in enumerate(grouped_packages)]
