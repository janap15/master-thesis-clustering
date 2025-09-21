import numpy as np

from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from data_models.package import Package
from services.cluster_manager import ClusterManager
from models.agglomerative_clustering import AgglomerativeClusteringModel
from models.kmeans_clustering import KMeansClusteringModel
from services.tomtom_client import TomTomClient


API_KEY = ""  # 🔹 replace with your real key
NUM_CLUSTERS = 3
NUM_PACKAGES = 15

def evaluate_clusters(distance_matrix, labels):
    distance_matrix = np.array(distance_matrix)
    sym_matrix = (distance_matrix + distance_matrix.T) / 2  # ensure symmetry

    # Silhouette can take a precomputed distance matrix
    silhouette = silhouette_score(sym_matrix, labels, metric="precomputed")

    # CH and DB need features, so we can still use MDS if needed
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    features = mds.fit_transform(sym_matrix)
    ch_score = calinski_harabasz_score(features, labels)
    db_score = davies_bouldin_score(features, labels)

    return {
        "silhouette": silhouette,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score
    }


# 🔹 Create dummy packages
def create_dummy_packages(seed=42):
    np.random.seed(seed)

    # Rough bounding box for Belgrade
    min_lat, max_lat = 44.75, 44.85
    min_lon, max_lon = 20.38, 20.50

    packages = []
    for i in range(NUM_PACKAGES):
        lat = np.random.uniform(min_lat, max_lat)
        lon = np.random.uniform(min_lon, max_lon)
        priority = np.random.randint(0, 4)  # priority 0-3
        pkg = Package(
            package_id=f"pkg_{i}",
            latitude=lat,
            longitude=lon,
            priority=priority
        )
        packages.append(pkg)

    return packages


def run_test():
    packages = create_dummy_packages()

    # 🔹 Use TomTom to get real distance matrix
    client = TomTomClient(API_KEY)
    job_id = client.submit_matrix_routing_request(packages)
    response = client.poll_matrix_routing_result(job_id)
    distance_matrix = client.response_to_result_matrix(response)

    print("=== Distance Matrix from TomTom (meters) ===")
    print(distance_matrix)

    # Agglomerative
    agg_model = AgglomerativeClusteringModel(n_clusters=2)
    cluster_manager_agg = ClusterManager(
        packages=packages,
        num_of_clusters=NUM_CLUSTERS,
        warehouse="W1",
        clustering_model=agg_model,
        tomtom_client=client,
    )
    cluster_manager_agg.distance_matrix = distance_matrix
    cluster_manager_agg.build_clusters()

    print("\n=== Agglomerative Clustering ===")
    for cluster in cluster_manager_agg.clusters:
        print(cluster)

    labels_agg = [p.get_cluster() for p in cluster_manager_agg.packages]
    scores_agg = evaluate_clusters(cluster_manager_agg.distance_matrix, labels_agg)
    print("\n=== Agglomerative Metrics ===")
    print(scores_agg)

    # KMeans
    kmeans_model = KMeansClusteringModel(n_clusters=2)
    cluster_manager_km = ClusterManager(
        packages=packages,
        num_of_clusters=NUM_CLUSTERS,
        warehouse="W1",
        clustering_model=kmeans_model,
        tomtom_client=client,
    )
    cluster_manager_km.distance_matrix = distance_matrix
    cluster_manager_km.build_clusters()

    print("\n=== KMeans Clustering ===")
    for cluster in cluster_manager_km.clusters:
        print(cluster)

    # KMeans
    labels_km = [p.get_cluster() for p in cluster_manager_km.packages]
    scores_km = evaluate_clusters(cluster_manager_km.distance_matrix, labels_km)
    print("\n=== KMeans Metrics ===")
    print(scores_km)

if __name__ == "__main__":
    run_test()
