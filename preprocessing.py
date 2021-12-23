import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

'''
Function:
    This function takes as input the parameters:
        -X=array of univariate time series X=(x_0, x_1, ..., x_N-1) of size=N
        -k=size of windows to be created
        -d=size of future days to average
    At first it creates (N-k+1) windows: (x_i, ..., x_i+k) for every i in {0, N-k}.
    Each window is an instance. For each window/instance:    
        -input x= (x_0, ..., x_k-d)
        -output y=0 or y=1 according to the labelling procedure:
        Calculate the mean of the remaining values (x_k-d+1, ..., x_k) and compare it 
        with the last value of input x.
            -if future_mean < x_k-d then y=0 (which indicates a downtrend in the future)
            -if future_mean > x_k-d then y=1 (which indicates an uptrend in the future)
'''


def input_output_split(X, k, d):

    total_instances = X.shape[0]

    # initialize N-k+1 windows of length: window_length
    number_of_windows = total_instances - k + 1
    window = np.empty([number_of_windows, k])

    # split the time series into windows
    for shift in range(number_of_windows):
        window[shift] = X[shift:shift + k]

    # first k-d values of every instance is the input window
    x = np.array(window[:, 0:k - d])

    # the remaining d values is the output window from which we extract label y
    y_window = np.array(window[:, k - d:])
    y = np.zeros(number_of_windows)
    for i in range(number_of_windows):
        if y_window[i].mean() > x[i, k - d - 1]:
            y[i] = 1

    return x, y


def candle_features_targets_split(X, k, d):
    X.reset_index(level=0, inplace=True)
    total_instances = X.shape[0]
    number_of_windows = total_instances - k + 1
    window = np.empty([number_of_windows, k, 5])

    # split the time series into windows window[shift][index][column]
    # x_3 in 3rd window = window[2][2][:] <-- last : indicated all columns close,open etc
    for shift in range(number_of_windows):
        window[shift][:][0] = X['Close'][shift:shift + k]
        window[shift][:][1] = X['Open'][shift:shift + k]
        window[shift][:][2] = X['High'][shift:shift + k]
        window[shift][:][3] = X['Low'][shift:shift + k]
        window[shift][:][4] = X['Volume'][shift:shift + k]

    x = np.array(window[:, 0:k - d, :])

    # the remaining d values is the output window from which we extract label y
    y_window = np.array(window[:, k - d:, 0])
    y_mean = np.empty(total_instances)
    for i in range(number_of_windows):
        y_mean[i] = y_window[i].mean()

    y = np.zeros(number_of_windows)
    for i in range(number_of_windows):
        if y_mean[i] > x[i, k - d - 1, 0]:
            y[i] = 1

    return x, y


def normalize(x):
    x_norm = np.empty([x.shape[0], x.shape[1]])
    for i in range(x.shape[0]):
        standard_deviation = x[i].std()
        mean = x[i].mean()
        if standard_deviation==0:
            x_norm[i].fill(0)
        else:
            for j in range(x.shape[1]):
                x_norm[i][j] = (x[i][j] - mean) / standard_deviation

    return x_norm


def grid_search(X_train, X_test, y_train, y_test):
    tuned_parameters = [
        {"kernel": ["rbf"],
         "gamma": [1e-3, 1e-4, 1e-5],
         "C": [1, 2, 5, 10]}]

    scores = ["precision", "recall"]

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