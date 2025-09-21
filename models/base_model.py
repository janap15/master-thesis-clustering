from abc import ABC, abstractmethod

class BaseClusteringModel(ABC):
    @abstractmethod
    def fit(self, similarity_matrix):
        pass