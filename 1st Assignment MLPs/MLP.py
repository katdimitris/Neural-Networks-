import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import preprocessing as prep

# device config
class MLP():

    def __init__(self, input_size, hidden_size=100, num_classes=2, epochs=10, bs=32, lr=0.01, device='cuda'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = bs
        self.learning_rate = lr

    def fit(self):
        dataset_obj = CryptoDataset(crypto=['BTC'], input_window_size=48, output_window_size=12)
        dataset = dataset_obj.get_dataset()
        train_dataset = dataset[:20000]
        test_dataset = dataset[20001:]

        print(train_dataset)
        print(test_dataset)

class CryptoDataset(Dataset):

    def __init__(self, crypto=None, input_window_size=48, output_window_size=12):
        # load dataset
        if crypto is None:
            crypto = ['BTC']
        x, y = prep.load_crypto_dataset(crypto, input_window_size, output_window_size)

        # convert x,y from lists to numpy array and then to torch objects
        self.x = torch.from_numpy(np.array(x))
        self.y = torch.from_numpy(np.array(y))

        # set the length of training samples
        self.n_samples = self.x.shape[0]

    def get_dataset(self):
        return self.x, self.y

    def get_item(self, index):
        return self.x[index], self.y[index]

    def get_length(self):
        return self.n_samples
