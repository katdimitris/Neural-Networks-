import time

import torch
from torch import nn

import MLP
import numpy as np
from sklearn.model_selection import train_test_split

import MLP
import preprocessing as prep

def main():
    # start timer
    t0 = time.time()

    X, y = prep.load_crypto_dataset()
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    print(X_train.shape[0])
    model = MLP.MLP_classifier()
    model.fit(X_train, y_train)
    model.predict(X_test, y_test)


    t1 = time.time()
    total = t1 - t0
    print("Total time: ", total)


if __name__ == "__main__":
    main()