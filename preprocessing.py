import numpy as np
import pandas as pd


def load_crypto_dataset(crypto=None, input_window_size=48, output_window_size=12):
    if crypto is None:
        crypto = ['BTC']

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
        x, y = input_output_split(X=data['close'], k=total_window_size, d=output_window_size)
        x_norm = normalize(x)

        # class distribution for each cryptocurrency pair
        print(f"{pair} samples in class -1: ", np.count_nonzero(y == -1))
        print(f"{pair} samples in class 1: ", np.count_nonzero(y == 1))

        input_arrays.append(x_norm)
        output_arrays.append(y)

    inputs_merged = []
    outputs_merged = []

    for i in range(0, len(input_arrays)):
        inputs_merged.extend(input_arrays[i])
        outputs_merged.extend(output_arrays[i])

    return inputs_merged, outputs_merged


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
    y = np.ones(number_of_windows)
    for i in range(number_of_windows):
        if y_window[i].mean() <= x[i, k - d - 1]:
            y[i] = -1

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
Function: normalize
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

