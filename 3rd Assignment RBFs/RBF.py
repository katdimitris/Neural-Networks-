import numpy as np
import kmeans
from scipy import linalg


def rbf(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (kmeans.euclidean_distance(x,c) ** 2))

def rbf(x, ti, sigma):
    return np.exp(-1 / (2 * sigma ** 2) * (kmeans.euclidean_distance(x,ti) ** 2))

class RBFN(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=10, rbf=rbf):
        self.k = k
        self.rbf = rbf
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):

        # compute hidden layer's centers and stds using kmeans algorithm
        k_means = kmeans.Kmeans(self.k)
        k_means.fit(X)
        self.centers = k_means.get_centroids()
        self.stds = k_means.get_clusters_std()

        # construct hidden layer matrix F
        self.F = np.empty([X.shape[0], self.k])
        for training_sample_idx in range(X.shape[0]):
            for hidden_unit_idx in range(self.k):
                self.F[training_sample_idx][hidden_unit_idx] = rbf(X[training_sample_idx],
                                                                   self.centers[hidden_unit_idx],
                                                                   self.stds[hidden_unit_idx])

        F = np.array(self.F)
        F_pseudoinverse = np.linalg.pinv(F)
        self.w = np.dot(F_pseudoinverse, y)
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
                                                                   self.stds[hidden_unit_idx])

        # predict class d (based on matrix equation: dot(F,weights)=d)
        for testing_sample_idx in range(X.shape[0]):
            d_pred.append(np.dot(F[testing_sample_idx],self.w))

        y_pred = np.ones(len(d_pred))
        for i in range(len(d_pred)):
            if d_pred[i] <= 0:
                y_pred[i] = -1

        return y_pred
