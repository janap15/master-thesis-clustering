from sklearn.cluster import AgglomerativeClustering
from models.base_model import BaseClusteringModel

class AgglomerativeClusteringModel(BaseClusteringModel):
    def __init__(self, n_clusters, linkage="complete"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None

    def fit(self, similarity_matrix):
        self.model = AgglomerativeClustering(
            metric="precomputed",
            n_clusters=self.n_clusters,
            linkage=self.linkage,
        )
        return self.model.fit(similarity_matrix).labels_