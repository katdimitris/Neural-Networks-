import time
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.hybrid import HIVECOTEV1

import preprocessing as prep
import RBF


def main():
    # first try on btc
    # start timer
    t0 = time.time()

    input_window_size = 48
    output_window_size = 12
    list_of_crypto = ['BTC']

    X, y = prep.load_crypto_dataset(list_of_crypto, input_window_size, output_window_size)

    X = np.array(X)
    y = np.array(y)

    idx = np.random.choice(len(X), 5000, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.4, shuffle=False)

    # Define and fit model
    # model = RBF.RBFN(k=30, distance_metric='euclidean')
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    model = KNeighborsTimeSeriesClassifier(distance='euclidean')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    t1 = time.time()
    total = t1 - t0
    print("Execution time: ", "{:.2f}".format(total), "seconds")


if __name__ == "__main__":
    main()


# example of kmeans with 3d points clustering
def test_example():
    np.random.seed(10)

    num_points = 500
    coords_x = np.random.randint(0, 50, num_points)
    coords_y = np.random.randint(0, 50, num_points)
    coords_z = np.random.randint(0, 50, num_points)

    labels = []
    for i in range(num_points):
        cond1 = (0 < coords_x[i] < 30 and 0 < coords_y[i] < 30 and 0 < coords_z[i] < 35)
        cond2 = (15 < coords_x[i] < 50 and 20 < coords_y[i] < 50 and 15 < coords_z[i] < 50)
        if cond1 or cond2:
            labels.append(1)
        else:
            labels.append(-1)

    X = np.array([coords_x, coords_y, coords_z])
    X = X.T

    y = np.array(labels)

    print("X shape is: ", X.shape)
    print("y shape is: ", y.shape)

    print("Points belonging to class 0: ", np.count_nonzero(y == -1))
    print("Points belonging to class 1: ", np.count_nonzero(y == 1))

    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, shuffle=False)
    model = RBF.RBFN(k=10)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("y pred is: ", y_pred)
    print("y true is: ", y_test)
    print(classification_report(y_test, y_pred))
