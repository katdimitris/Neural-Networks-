import numpy as np
from sktime.distances import _dtw


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# implementation of dtw algorithm
def dtw_distance(x1, x2):
    length = len(x1)
    dtw_matrix = np.zeros((length + 1, length + 1))
    for i in range(length + 1):
        for j in range(length + 1):
            dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0
    for i in range(1, length + 1):
        for j in range(1, length + 1):
            cost = abs(x1[i - 1] - x2[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[length, length]


class Distances:
    def __init__(self, distance_metric='euclidean_distance'):
        self.distance_metric = distance_metric

    def get_distance(self, x1, x2):

        if self.distance_metric == 'euclidean':
            return euclidean_distance(x1, x2)

        # dtw usage of sktime distance (kinda works)
        if self.distance_metric == 'dtw':
            x1 = x1.reshape((x1.shape[0], 1))
            x2 = x2.reshape((x2.shape[0], 1))
            obj = _dtw._DtwDistance()
            dist = obj.distance(x1, x2)
            return dist
