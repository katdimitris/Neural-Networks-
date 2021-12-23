# import python libraries
import time

import numpy as np
import pandas as pd

# import sklearn packages
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC

import preprocessing
import plots


# USING YAHOO API
# data = web.DataReader('BTC-USD', 'yahoo')
# data.drop('Adj Close', axis=1, inplace=True)
# data.dropna(inplace=True)

def grid_search(X_train, X_test, y_train, y_test):
    tuned_parameters = [
        {"kernel": ["rbf"],
         "gamma": [0.2, 0.1, 0.05, 0.01, 0.001],
         "C": [1, 5, 10]}]

    scores = ["f1"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, scoring="%s_macro" % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    return clf.best_params_


def main():
    t0 = time.time()
    input_window_size = 48
    output_window_size = 10
    total_window_size = input_window_size + output_window_size

    cryptocurrencies = ['ETH']
    currency = 'USDT'
    path = 'C:/Users/Dimitris/Desktop/NNDL/hourly_binance/'
    pairs = []

    for crypto in cryptocurrencies:
        pairs.append(f'{crypto}{currency}')

    input_arrays = []
    output_arrays = []

    for pair in pairs:
        # data loading
        data = pd.read_feather(path + f'{pair}.feather')
        data.set_index('date', inplace=True)
        data.dropna(inplace=True)
        x, y = preprocessing.input_output_split(X=data['close'], k=total_window_size, d=output_window_size)
        x_norm = preprocessing.normalize(x)

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

        # print("TOTAL 0: ", outputs_merged.count(0))
        # print("TOTAL 1: ", outputs_merged.count(1))

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(inputs_merged, outputs_merged, test_size=0.3, shuffle=False)

    tuned_parameters = grid_search(X_train, X_test, y_train, y_test)

    # clf = svm.SVC(C=5, gamma=0.01, class_weight='balanced')  # kernel=rbf default value
    # clf = clf.fit(X_train, y_train)
    #
    # # Predict the test set
    # y_pred = clf.predict(X_test)
    # print("---------------------------------------------------")
    # print("INPUT SIZE: ", input_window_size)
    # print("OUTPUT SIZE: ", output_window_size)
    # print("RESULTS:")
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))  # Output
    # print("---------------------------------------------------")
    # print()
    # t1 = time.time()
    # total = t1 - t0
    # print("Total time: ", total)

if __name__ == "__main__": main()







