import time
import MLP
import numpy as np
from sklearn.model_selection import train_test_split

import preprocessing as prep

def main():
    # start timer
    t0 = time.time()

    dataset_obj = MLP.CryptoDataset(crypto=['BTC'], input_window_size=48, output_window_size=12)
    dataset = dataset_obj.get_dataset()
    train_dataset = dataset[:int(dataset_obj.get_length() * 0.8)]
    test_dataset = dataset[int(dataset_obj.get_length() * 0.8) + 1:]
    x_train, y_train = train_dataset
    #x_test, y_test = test_dataset
    print(x_train.shape)
   # print(x_test.shape)
    # Split into train and test
   # X_train, X_test, y_train, y_test = train_test_split(inputs_merged, outputs_merged, test_size=0.3, shuffle=False)



    t1 = time.time()
    total = t1 - t0
    print("Total time: ", total)


if __name__ == "__main__":
    main()