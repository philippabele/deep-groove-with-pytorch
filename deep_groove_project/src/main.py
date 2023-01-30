import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from src.FeatureDataset import FeatureDataset
from src.NeuralNetwork import NeuralNetwork

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    print(model)

    #  Loading and preparing Dataframe
    dataframe = pd.read_csv("../data/first_dataset.csv")

    dataframe['Lifetime'].where(dataframe['Lifetime'] >= 8760, 1, inplace=True)
    dataframe['Lifetime'].where(dataframe['Lifetime'] < 8760, 0, inplace=True)

    # Creating custom Dataset
    train_dataset = FeatureDataset(dataframe[:int(len(dataframe)/2)])
    test_dataset = FeatureDataset(dataframe[int(len(dataframe)/2):])

    print(f"length of Datasets - training: {len(train_dataset)}, test: {len(test_dataset)}")

    # Parameters
    params = {'batch_size': 20,
              'shuffle': True}

    # Creating the Dataloaders
    train_dataloader = DataLoader(train_dataset, **params)
    test_dataloader = DataLoader(test_dataset, **params)

    for batch in train_dataloader:
        print(f"Batch: {batch[0]}")
        break

    learning_rate = 1e-3
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 5
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # print(f"X: {X.dtype}")
        # print(f"y: {y.dtype}")
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
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


if __name__ == '__main__':
    main()
