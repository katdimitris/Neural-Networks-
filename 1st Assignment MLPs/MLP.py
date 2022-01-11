import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

import preprocessing as prep

''' ------------------------NOT WORKING------------------------------------- '''

class CryptoDataset(Dataset):

    def __init__(self, X, y):
        self.x = torch.from_numpy(X)
        self.x = self.x.float()
        self.y = torch.from_numpy(y)
        self.y = self.y.long()
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples




class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_epochs, batch_size,
                 learning_rate):
        super(FNN, self).__init__()  # Inherited from the parent class nn.Module
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.l1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
        self.l2 = nn.Linear(hidden_size, hidden_size)  # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # Forward pass: stacking each layer together
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

class MLP_classifier():

    def __init__(self, input_size=48, hidden_size=100, num_classes=2,
                 num_epochs=10, batch_size=32, learning_rate=0.001):
        self.FNN = FNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,
                       num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate



    def fit(self, X, y):

        self.train_dataset = CryptoDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.FNN.parameters(), lr=self.learning_rate)

        # training loop
        n_total_steps = len(train_loader)
        for epoch in range(self.num_epochs):
            for i, (input_windows, labels) in enumerate(
                    train_loader):  # Load a batch of images with its (index, data, class)

                optimizer.zero_grad()  # Intialize the hidden weight to all zeros
                outputs = self.FNN(input_windows)  # Forward pass: compute the output class given a image
                loss = criterion(outputs,
                                 labels)  # Compute the loss: difference between the output class and the pre-given label
                loss.backward()  # Backward pass: compute the weight
                optimizer.step()  # Optimizer: update the weights of hidden nodes

                if (i + 1) % 100 == 0:  # Logging
                    print(f'epoch {epoch+1} / {self.num_epochs}, step {i+1}/{n_total_steps}, loss={loss.item()}')

    def predict(self, X, y):
        self.test_dataset = CryptoDataset(X, y)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for input_windows, labels in test_loader:
                # labels = labels.to(device)
                outputs = self.FNN(input_windows)

                _, predictions = torch.max(outputs, 1)
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'accuracy = {acc}')
            print(classification_report(predictions, labels))

        return