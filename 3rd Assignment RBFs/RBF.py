import numpy as np
import kmeans
from scipy import linalg
from time_series_distance_metrics import Distances


def rbf(x, c, s, distance_metric):
    distance_obj = Distances(distance_metric)
    return np.exp(-1 / (2 * s ** 2) * distance_obj.get_distance(x, c) ** 2)


class RBFN(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=10, rbf=rbf, distance_metric='euclidean'):
        self.k = k
        self.rbf = rbf
        self.w = np.random.randn(k)
        # ωself.b = np.random.randn(1)
        self.distance_metric = distance_metric

    def fit(self, X, y):

        # compute hidden layer's centers and stds using k-means algorithm
        k_means = kmeans.Kmeans(k=self.k, distance_metric=self.distance_metric)
        k_means.fit(X)

        self.centers = k_means.get_centroids()
        self.stds = k_means.get_clusters_std()

        # construct hidden layer matrix F
        self.F = np.empty([X.shape[0], self.k])
        for training_sample_idx in range(X.shape[0]):
            for hidden_unit_idx in range(self.k):
                self.F[training_sample_idx][hidden_unit_idx] = rbf(X[training_sample_idx],
                                                                   self.centers[hidden_unit_idx],
                                                                   self.stds[hidden_unit_idx],
                                                                   self.distance_metric)

        F = np.array(self.F)
        F_pseudoinverse = np.linalg.pinv(F)
        self.w = np.matmul(F_pseudoinverse, y)
        # print(self.w)

    def predict(self, X):

        # initialize predictions as empty list
        d_pred = []

        # create hidden layer's rbf matrix for the given test samples to be predicted
        F = np.empty([X.shape[0], self.k])
        for testing_sample_idx in range(X.shape[0]):
            for hidden_unit_idx in range(self.k):
                F[testing_sample_idx][hidden_unit_idx] = rbf(X[testing_sample_idx],
                                                             self.centers[hidden_unit_idx],
                                                             self.stds[hidden_unit_idx],
                                                             self.distance_metric)

        # predict class d (based on matrix equation: dot(F,weights)=d)
        for testing_sample_idx in range(X.shape[0]):
            d_pred.append(np.dot(F[testing_sample_idx], self.w))

        y_pred = np.ones(len(d_pred))
        for i in range(len(d_pred)):
            if d_pred[i] <= 0:
                y_pred[i] = -1

        return y_pred
