import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()

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