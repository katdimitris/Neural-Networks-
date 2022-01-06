# import python libraries
import time

# import sklearn packages
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import preprocessing as prep
import svm


def main():
    # start timer
    t0 = time.time()

    # grid search to find best performance window sizes based on training set
    # input_window_sizes = [36, 48, 60]
    # output_window_sizes = [8, 12, 16]
    # for input_window_size in input_window_sizes:
    #     for output_window_size in output_window_sizes:

    # set initial parameters
    input_window_size = 48
    output_window_size = 12
    list_of_crypto = ['BTC']

    inputs_merged, outputs_merged = prep.load_crypto_dataset(list_of_crypto, input_window_size, output_window_size)

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
