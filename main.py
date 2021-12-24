# import python libraries
import time
import numpy as np
import pandas as pd

# import sklearn packages
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

import preprocessing as prep
import plots
import svm


def main():
    # start timer
    t0 = time.time()

    # grid search to find best performance window sizes based on training set
    # input_window_sizes = [36, 48, 60]
    # output_window_sizes = [8, 12, 16]
    # for input_window_size in input_window_sizes:
    #     for output_window_size in output_window_sizes:

    # set the optimal windows size
    input_window_size = 48
    output_window_size = 12
    total_window_size = input_window_size + output_window_size

    # Define the list of crypto to be analyzed and the currency
    crypto = ['BTC']
    currency = 'USDT'

    # define the path of the dataset
    path = 'C:/Users/Dimitris/Desktop/NNDL/hourly_binance/'

    # create cryptocurrency pairs
    pairs = []
    for c in crypto:
        pairs.append(f'{c}{currency}')

    input_arrays = []
    output_arrays = []

    for pair in pairs:
        # data loading
        data = pd.read_feather(path + f'{pair}.feather')
        data.set_index('date', inplace=True)
        data.dropna(inplace=True)
        x, y = prep.input_output_split(X=data['close'], k=total_window_size, d=output_window_size)
        x_norm = prep.normalize(x)

        # plots.plot_class_distribution(y)
        # class distribution for each coin
        print(f"{pair} 0: ", np.count_nonzero(y == 0))
        print(f"{pair} 1: ", np.count_nonzero(y == 1))

        input_arrays.append(x_norm)
        output_arrays.append(y)

    inputs_merged = []
    outputs_merged = []

    for i in range(0, len(input_arrays)):
        inputs_merged.extend(input_arrays[i])
        outputs_merged.extend(output_arrays[i])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(inputs_merged, outputs_merged, test_size=0.3, shuffle=False)

    ''' SUPPORT VECTOR MACHINE '''

    # Define and fit model
    clf = svm.SVC(C=5, class_weight='balanced', gamma=0.005)  # kernel=rbf default value
    clf = clf.fit(X_train, y_train)

    # Predict the test set
    y_pred = clf.predict(X_test)
    print("Training window size: ", input_window_size, "Prediction window size: ", output_window_size)
    # svm.grid_search(X_train, y_train, X_test, y_test)
    print(confusion_matrix(y_test, y_pred))
    #plots.plot_confusion_matrix(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    ''' K NEAREST NEIGHBORS ALGORITHM '''

    # # Define and fit model
    # clf = KNeighborsClassifier(n_neighbors=1000)
    # clf = clf.fit(X_train, y_train)
    #
    # # Predict the test set
    # y_pred = clf.predict(X_test)
    # print("Training window size: ", input_window_size, "Prediction window size: ", output_window_size)
    # # svm.grid_search(X_train, y_train, X_test, y_test)
    # print(confusion_matrix(y_test, y_pred))
    # plots.plot_confusion_matrix(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))


    ''' NEAREST CENTROID ALGORITHM '''

    # Define and fit model
    # clf = NearestCentroid()
    # clf = clf.fit(X_train, y_train)

    # # Predict the test set
    # y_pred = clf.predict(X_test)
    # print("Training window size: ", input_window_size, "Prediction window size: ", output_window_size)
    # # svm.grid_search(X_train, y_train, X_test, y_test)
    # print(confusion_matrix(y_test, y_pred))
    # plots.plot_confusion_matrix(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    t1 = time.time()
    total = t1 - t0
    print("Total time: ", total)


if __name__ == "__main__":
    main()
