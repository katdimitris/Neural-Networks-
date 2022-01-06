import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


'''
Function:
    Plots candle chart of data
Parameters:
    1) data: DataFrame with index='Date' and columns='Open','Close','High','Low'
'''
def plot_price_volume_chart(data):
    data = data.set_index('date')
    colors = mpf.make_marketcolors(up='#00ff00', down='#ff0000',
                                   wick='inherit',
                                   edge='inherit',
                                   volume='in')
    my_style = mpf.make_mpf_style(base_mpf_style='nightclouds')

    mpf.plot(data, type='candle', style=my_style, volume=True)


def plot_close_price(data):
    fig = plt.figure(figsize=(12, 5), dpi=100)
    print(data)
    sns.set()
    plt.plot(data, label="BTC 1H close price")
    plt.xlabel("Training samples")
    plt.ylabel("USDT")
    plt.title("BTC close price")
    plt.legend()
    plt.show()


def get_class_distribution(obj):
    count_dict = {
        'Class 0': 0,
        'Class 1': 0,
    }
    for i in obj:
        if i == 0:
            count_dict['Class 0'] += 1
        elif i == 1:
            count_dict['Class 1'] += 1

    return count_dict


def plot_total_class_distribution(y):
    class_0 = np.count_nonzero(y == 0)
    class_1 = np.count_nonzero(y == 1)
    data = {'Class_0': class_0, 'Class_1': class_1}
    classes = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 6), dpi=100)
    ax = sns.barplot(x=classes, y=values)
    ax.set(xlabel='Classes', ylabel='Value')
    ax.set_title("Class Distribution")
    plt.show()


def plot_train_test_class_distribution(y_train, y_test):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Train
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(),
                x="variable", y="value", hue="variable", ax=axes[0]).set_title('Class Distribution in Train Set',
                                                                               fontsize=16)

    # Test
    sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(),
                x="variable", y="value", hue="variable", ax=axes[1]).set_title('Class Distribution in Test Set',
                                                                               fontsize=16)
    axes[0].legend([], [], frameon=False)
    axes[1].legend([], [], frameon=False)
    plt.show()


def plot_window_size_first_confusion_matrix():
    training_window = [12,24,36,48,60,72]
    prediction_window = [4,8,12,16,24]

    #windows =[training_window][prediction_window]
    windows = [[0.58, 0.58, 0.57, 0.57, 0.57],
               [0.61, 0.62, 0.61, 0.61, 0.61],
               [0.63, 0.63, 0.64, 0.63, 0.62],
               [0.64, 0.65, 0.67, 0.65, 0.63],
               [0.65, 0.66, 0.66, 0.65, 0.66],
               [0.65, 0.66, 0.66, 0.65, 0.65]]
    df_windows = pd.DataFrame(windows, index=training_window,
                         columns=prediction_window)
    fig = plt.figure(figsize=(10, 7))

    fig = sns.heatmap(df_windows, linewidths=0.01, cmap="plasma", annot=True)
    #fig.set(xlabel='Prediction window size', ylabel='Training window size', title='Benchmark SVM Accuracy VS Window Sizes')
    plt.xlabel("Prediction window size in hours",fontsize=14)
    plt.ylabel("Training window size in hours",fontsize=14)
    plt.title("Benchmark SVM Accuracy VS Window Sizes",fontsize=16)  # You can comment this line out if you don't need title
    plt.show()


def plot_window_size_second_confusion_matrix():
    training_window = [36, 48, 60]
    prediction_window = [8, 12, 16]

    # windows =[training_window][prediction_window]
    windows = [[0.52, 0.53, 0.52],
               [0.54, 0.55, 0.54],
               [0.53, 0.53, 0.52]]
    df_windows = pd.DataFrame(windows, index=training_window,
                              columns=prediction_window)
    fig = plt.figure(figsize=(8, 5), dpi=150)

    fig = sns.heatmap(df_windows, linewidths=0.01, cmap="plasma", annot=True)
    plt.xlabel("Prediction window size in hours", fontsize=12)
    plt.ylabel("Training window size in hours", fontsize=12)
    plt.title("SVM Accuracy VS Window Sizes (tuned C and gamma)", fontsize=14)
    plt.show()


def plot_grid_search_confusion_matrix():
    C_values = [5, 10, 15, 20]
    gamma_values = [0.2, 0.1, 0.05, 0.01, 0.005]

    # windows =[training_window][prediction_window]
    parameters = [[.525, .532, .530, .548, .549],
               [.524, .528, .525, .542, .548],
               [.523, .525, .525, .540, .547],
               [.523, .526, .523, .539, .547]]
    df_parameters = pd.DataFrame(parameters, index=C_values,
                              columns=gamma_values)
    fig = plt.figure(figsize=(8, 5), dpi=150)

    fig = sns.heatmap(df_parameters, linewidths=0.01, cmap="plasma", annot=True)
    plt.xlabel("Gamma values", fontsize=12)
    plt.ylabel("C values", fontsize=12)
    plt.title("SVM hyper parameters tuning", fontsize=14)
    plt.show()


def  plot_confusion_matrix(confusion_matrix):

    fig = plt.figure(figsize=(6, 4), dpi=150)

    fig = sns.heatmap(confusion_matrix, linewidths=0.01, annot=True)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.title("Test set Confusion Matrix",
              fontsize=14)  # You can comment this line out if you don't need title
    plt.show()
