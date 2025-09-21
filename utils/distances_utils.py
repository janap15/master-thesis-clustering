import numpy as np

def normalize_matrix(matrix):
    return matrix/np.max(matrix)

def get_priority_diversity_matrix(priorities):
    priority_diversity_matrix  = np.abs(np.subtract.outer(priorities, priorities))
    max_priority_matrix  = np.max(priority_diversity_matrix)
    return 1 - priority_diversity_matrix / max_priority_matrix if max_priority_matrix > 0 else np.ones_like(priority_diversity_matrix)
