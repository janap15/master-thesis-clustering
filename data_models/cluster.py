class Cluster:

    def __init__(self, cluster_id, packages, warehouse):
        self.cluster_id = cluster_id
        self.packages = packages
        self.warehouse = warehouse

    def get_id(self):
        return self.cluster_id

    def set_id(self, cluster_id):
        self.cluster_id = cluster_id

    def get_packages(self):
        return self.packages

    def add_package(self, package):
        self.packages.append(package)

    def add_packages(self, packages):
        self.packages.extend(packages)

    def get_warehouse(self):
        return self.warehouse

    def set_warehouse(self, warehouse):
        self.warehouse = warehouse

    def set_package_cluster_at_index(self, index, cluster_id):
        self.packages[index].set_cluster(cluster_id)

    def get_cluster_size(self):
        return len(self.packages)

    def remove_package_at_index(self, index):
        self.packages.pop(index)

    def remove_package(self, package):
        self.packages.remove(package)

    def count_packages_by_priority(self):
        priority_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for package in self.packages:
            priority_counts[package.get_priority()] += 1
        return priority_counts

    # each cluster has the same start and destination point, which is warehouse
    # add them for the purpose of sending requests to th Waypoint Optimization endpoint
    def create_waypoints(self):
        return [self.warehouse] + self.packages + [self.warehouse]

    def __str__(self):
        data = f"Cluster {self.cluster_id}\n"
        for package in self.packages:
            data += f"{package}\n"
        return data
