import matplotlib.pyplot as plt
import numpy as np


# euclidean distance of two vectors x1, x2
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


'''
Class kmeans: (unsupervised clustering algorithm)
Clusters a unlabeled dataset into k different clusters 
Each data point is assigned to the cluster with the nearest mean
'''


class Kmeans:
    def __init__(self, K=5, max_iters=100, const_std=False):

        self.K = K
        self.max_iters = max_iters
        self.const_std = const_std

        # create an empty list representing the K clusters to be created and filled with data poits
        self.clusters = [[] for _ in range(self.K)]

        # create an empty list representing the centers of the K clusters
        self.centroids = []


    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize the K centers with distinct random data points
        self.centroids = self.X[np.random.choice(self.n_samples, self.K, replace=False)]

        # Optimize clusters
        for _ in range(self.max_iters):

            # Assign samples to closest centroids (create clusters)
            self._update_clusters()

            # Store old centroids and calculate new centroids resulting from cluster update
            old_centroids = self.centroids.copy()
            self._update_centroids()

            # check if clusters have changed
            if self._converged(old_centroids):
                break

        # plot the created clusters
        #self.plot()
            # Classify samples as the index of their clusters
        return


    def predict(self, X):
        predictions = []
        for idx, sample in enumerate(X):
            predictions.append(self._closest_centroid(sample))
        self.plot()
        return predictions


    def get_centroids(self):
        return self.centroids


    def get_clusters_std(self):
        stds = []
        for cluster in self.clusters:
            if np.std(cluster) == 0:
                stds.append(0.01)
            else:
                stds.append(np.std(cluster))
        stds = np.array(stds)
        return stds


    def _update_clusters(self):
        # empty clusters
        self.clusters = [[] for _ in range(self.K)]
        # Assign each sample in X to the cluster with the smallest centroid distance
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample)
            self.clusters[centroid_idx].append(idx)


    def _closest_centroid(self, sample):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        closest_index = np.argmin(distances)
        return closest_index


    def _update_centroids(self):
        for cluster_idx, cluster in enumerate(self.clusters):
            self.centroids[cluster_idx] = np.mean(self.X[cluster], axis=0)


    def _converged(self, old_centroids):
        if np.sum(old_centroids != self.centroids) != 0:
            return False
        return True

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
