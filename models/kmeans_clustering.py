import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from models.base_model import BaseClusteringModel

class KMeansClusteringModel(BaseClusteringModel):
    def __init__(self, n_clusters, n_components=2, random_state=42):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.model = None

    def fit(self, similarity_matrix):
        # Ensure matrix is symmetric
        if not np.allclose(similarity_matrix, similarity_matrix.T):
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

        # Use MDS to project distances → feature space
        mds = MDS(
            n_components=self.n_components,
            random_state=self.random_state,
            dissimilarity="precomputed"
        )
        features = mds.fit_transform(similarity_matrix)

        # Apply KMeans on feature space
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return self.model.fit(features).labels_