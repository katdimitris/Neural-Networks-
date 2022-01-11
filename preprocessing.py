import numpy as np
import pandas as pd

'''
Function: load_crypto_dataset
    This function takes as input the parameters:
        -crypto: list of crypto names to be loaded from the csv file (default value=['BTC'])
        -input_window_size: the dimension of input size (default value = 48 hours)
        -output_window_size: price duration (measured in hours) to be averaged in order to 
         define an up or down label(default value = 12 hours)
    The function loads hourly crypto prices and splits them into time series samples of size=input_window_size.
    Then it labels them according to the function 'input_output_split'.
    It returns all input vectors X[i], and their binary label y[i].
'''
def load_crypto_dataset(crypto=None, input_window_size=48, output_window_size=12):
    if crypto is None:
        crypto = ['BTC']

    total_window_size = input_window_size + output_window_size

    # Define the list of crypto to be analyzed and the currency
    currency = 'USDT'

    # define the path of the dataset
    path = 'C:/Users/Dimitris/Desktop/NNDL/hourly_binance/'

    # create cryptocurrency pairs
    pairs = []
    for c in crypto:
        pairs.append(f'{c}{currency}')

    X_pairs = []
    y_pairs = []

    for pair in pairs:
        # data loading
        data = pd.read_feather(path + f'{pair}.feather')
        data.set_index('date', inplace=True)
        data.dropna(inplace=True)
        x_pair, y_pair = input_output_split(X=data['close'], k=total_window_size, d=output_window_size)
        x_pair_norm = normalize(x_pair)

        # class distribution for each cryptocurrency pair
        print(f"{pair} samples in class -1: ", np.count_nonzero(y_pair == -1))
        print(f"{pair} samples in class 1: ", np.count_nonzero(y_pair == 1))

        X_pairs.append(x_pair_norm)
        y_pairs.append(y_pair)

    X = []
    y = []

    for i in range(0, len(X_pairs)):
        X.extend(X_pairs[i])
        y.extend(y_pairs[i])

    return X, y


'''
Function: input_output_split
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
    y = np.ones(number_of_windows)
    for i in range(number_of_windows):
        if y_window[i].mean() <= x[i, k - d - 1]:
            y[i] = 0

    return x, y


'''
Function: candle_features_targets_split
'''
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

'''
Function: Takes as input a time series and it normalizes it 
'''
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

