import sys
import matplotlib.pyplot as plt
import pandas_datareader as web
import mplfinance as mpf
import numpy as np
import pandas as pd
import datetime as dt

from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

import preprocessing as prep
'''
Function:
    Plots candle chart of data
Parameters:
    1) data: DataFrame with index='Date' and columns='Open','Close','High','Low'
'''
def plot_price_volume_chart(data):
    colors = mpf.make_marketcolors(up='#00ff00',
                                   down='#ff0000',
                                   wick='inherit',
                                   edge='inherit',
                                   volume='in')
    my_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=colors)

    mpf.plot(data, type='candle', style=my_style, volume=True)



data = web.DataReader('BTC-USD', 'yahoo')
data.drop('Adj Close', axis=1, inplace=True)
data.dropna(inplace=True)

# set the length of the windows
input_window_size = 4
output_window_size = 1
total_window_size = input_window_size + output_window_size

x, y = prep.univariate_time_series_preprocessing(X=data['Close'], k=total_window_size, d=output_window_size)

print("1: ", np.count_nonzero(y == 1))
print("0: ", np.count_nonzero(y == 0))

x_norm = prep.univariate_normalize(x)

X_train, X_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, shuffle=False)

# scaler = MinMaxScaler()
# norm = scaler.fit(X_train)
# X_train_norm = norm.transform(X_train)
# X_test_norm = norm.transform(X_test)

clf = svm.SVC(kernel='rbf')
clf = clf.fit(X_train, y_train)

# Predict the test set
y_pred = clf.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))#Output









