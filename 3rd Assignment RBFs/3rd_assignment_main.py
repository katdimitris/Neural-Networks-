import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import preprocessing as prep
import RBF
from sktime.distances._dtw import _DtwDistance

import dtw


def distance_test():
    t0 = time.time()

    input_window_size = 48
    output_window_size = 12
    list_of_crypto = ['BTC']
    X, y = prep.load_crypto_dataset(list_of_crypto, input_window_size, output_window_size)

    X = np.array(X)
    idx = np.random.choice(len(X), size=200, replace=False)
    X_sample = X[idx]

    x1 = X_sample[0]
    x1 = x1.reshape((x1.shape[0], 1))

    x2 = X_sample[1]
    x2 = x2.reshape((x2.shape[0], 1))

    dist1, _, _, _ = dtw.dtw(X_sample[0], X_sample[1], dist='euclidean_distance')

    obj = _DtwDistance()
    dist2 = obj.distance(x1,x2)

    t1 = time.time()
    total = t1 - t0
    print("Total time: ", total)


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

    # idx = np.random.choice(len(X), size=500, replace=False)
    # X_sample = X[idx]
    # y_sample = y[idx]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


    # Define and fit model
    model = RBF.RBFN(k=100, distance_metric='euclidean')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Predict the test set
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    '''K FOLDS CROSS VALIDATION '''
    # splits = 5
    # kf = KFold(n_splits=splits, shuffle=False)
    # model = RBF.RBFN(k=100, distance_metric='euclidean')
    # acc_score=[]
    #
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index,:], X[test_index,:]
    #     y_train, y_test = y[train_index], y[test_index]
    #     model.fit(X_train,y_train)
    #     pred_values = model.predict(X_test)
    #
    #     acc = accuracy_score(pred_values, y_test)
    #     acc_score.append(acc)
    #
    # avg_acc_score = sum(acc_score) / splits
    #
    # print('accuracy of each fold - {}'.format(acc_score))
    # print('Avg accuracy : {}'.format(avg_acc_score))

    t1 = time.time()
    total = t1 - t0
    print("Total time: ", total)


if __name__ == "__main__":
    main()
    # distance_test()


# example with 3d points
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
