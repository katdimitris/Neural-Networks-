import time
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import MLP
import preprocessing as prep
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

def main():
    # start timer
    t0 = time.time()

    X, y = prep.load_crypto_dataset(crypto=['BTC'])
    X = np.array(X)
    y = np.array(y)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)


    model = MLP.MLP_classifier(num_epochs=100, hidden_size=192, batch_size=32, learning_rate=0.002)

    model.fit(X_train, y_train, X_val, y_val)

    t1 = time.time()
    total = t1 - t0
    print("Training time: ", total)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


    t2 = time.time()
    total = t2 - t0
    print("Total time: ", total)


if __name__ == "__main__":
    main()