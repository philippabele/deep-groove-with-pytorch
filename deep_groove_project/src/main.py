import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from src.FeatureDataset import FeatureDataset
from src.NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    print(model)

    #  Loading and preparing Dataframe
    dataframe = pd.read_csv("../data/first_dataset.csv")

    # shuffle Dataframe
    dataframe = dataframe.sample(frac=1)

    print(dataframe.head())
    dataframe['Label'] = 0
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760, 1, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*2, 2, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*3, 3, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*4, 4, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*4, 5, inplace=True)

    print(dataframe.head())
    # Creating custom Dataset
    train_dataset = FeatureDataset(dataframe[:int(len(dataframe)/2)])
    test_dataset = FeatureDataset(dataframe[int(len(dataframe)/2):])

    print(f"length of Datasets - training: {len(train_dataset)}, test: {len(test_dataset)}")

    # Parameters
    params = {'batch_size': 10,
              'shuffle': True}

    # Creating the Dataloaders
    train_dataloader = DataLoader(train_dataset, **params)
    test_dataloader = DataLoader(test_dataset, **params)

    for batch in train_dataloader:
        print(f"Batch: {batch[0]}")
        break

    learning_rate = 1e-7
    print(f"LR:  {learning_rate}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 200

    accuracy, avg_loss = [], []
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn, [accuracy, avg_loss])
    print("Done!")
    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy in %")
    plt.show()
    plt.plot(avg_loss)
    plt.xlabel("Epoch")
    plt.ylabel("AVG Loss")
    plt.show()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>4d}/{size:>4d}]")


def test_loop(dataloader, model, loss_fn, charts):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    charts[0].append((100 * correct))
    charts[1].append(test_loss)

if __name__ == '__main__':
    main()
