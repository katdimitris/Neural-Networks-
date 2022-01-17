import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import pandas as pd


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)



class MLP_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_epochs, batch_size,
                 learning_rate):
        super(MLP_model, self).__init__()  # Inherited from the parent class nn.Module
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.l1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.l2 = nn.Linear(hidden_size, hidden_size)  # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.l4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # Forward pass: stacking each layer together
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)

        return out

class MLP_classifier():

    def __init__(self, input_size=48, hidden_size=100, num_classes=2,
                 num_epochs=100, batch_size=32, learning_rate=0.01):
        self.model= MLP_model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,
                       num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.device = 'cuda:0'
        self.model.to(self.device)


    def fit(self, X_train, y_train, X_val, y_val):
        train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Define dictionaries to store accuracy/epoch and loss/epoch for trainset
        accuracy_stats = {
            'train': [],
            "val": []
        }
        loss_stats = {
            'train': [],
            "val": []
        }

        print("Begin training.")
        for e in range(1, self.num_epochs + 1):

            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            self.model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                optimizer.zero_grad()
                y_train_pred = self.model(X_train_batch)

                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            # VALIDATION
            with torch.no_grad():

                val_epoch_loss = 0
                val_epoch_acc = 0

                self.model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_pred = self.model(X_val_batch)

                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
            loss_stats['train'].append(train_epoch_loss / len(train_loader))
            loss_stats['val'].append(val_epoch_loss / len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

            print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | '
                  f'Val Loss: {val_epoch_loss / len(val_loader):.5f} | '
                  f'Train Acc: {train_epoch_acc/len(train_loader):.3f}| '
                  f'Val Acc: {val_epoch_acc/len(val_loader):.3f}')


        # Create dataframes
        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
            columns={"index": "epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
            columns={"index": "epochs"})

        # Plot the dataframes
        sns.set()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=200)
        fig.suptitle(f'Epochs={self.num_epochs}, '
                    f'LR={self.learning_rate}, '
                    f'BS={self.batch_size}', fontsize=14)

        sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable",
                         ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
        sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable",
                         ax=axes[1]).set_title('Train-Val Loss/Epoch')
        plt.show()


    def predict(self, X_test):
        y_test = np.empty(X_test.shape[0])
        test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
        y_pred_list = []
        with torch.no_grad():
            self.model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                _, y_pred_tags = torch.max(y_test_pred, dim=1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

        return y_pred_list
