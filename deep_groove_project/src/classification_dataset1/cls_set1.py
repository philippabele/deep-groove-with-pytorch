import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from classification_dataset1.FeatureDataset import FeatureDataset
from classification_dataset1.NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt


def cls_set1(params=None, output={"cli": True, "plot": True}):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    if output["cli"]:
        print(model)

    #  Loading and preparing Dataframe
    dataframe = pd.read_csv("../../dataset/dataset1.csv")

    # shuffle Dataframe
    dataframe = dataframe.sample(frac=1)

    if output["cli"]:
        print(dataframe.head())
    dataframe['Label'] = 0
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760, 1, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*2, 2, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*3, 3, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*4, 4, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*4, 5, inplace=True)

    if output["cli"]:
        print(dataframe.head())
    # Creating custom Dataset
    train_dataset = FeatureDataset(dataframe[:int(len(dataframe)/2)])
    test_dataset = FeatureDataset(dataframe[int(len(dataframe)/2):])

    if output["cli"]:
        print(f"length of Datasets - training: {len(train_dataset)}, test: {len(test_dataset)}")

    # Parameters
    if params is None:
        params = {'batch_size': 10,
                  'shuffle': True,
                  'learning_rate': 1e-6,
                  'epochs': 200}

    # Creating the Dataloaders
    train_dataloader = DataLoader(train_dataset, shuffle=params["shuffle"], batch_size=params["batch_size"])
    test_dataloader = DataLoader(test_dataset, shuffle=params["shuffle"], batch_size=params["batch_size"])
    if output["cli"]:
        for batch in train_dataloader:
            print(f"Batch: {batch[0]}")
            break
        print(f"LR:  {params['learning_rate']}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

    accuracy, avg_loss = [], []
    for e in range(params['epochs']):
        if output["cli"]:
            print(f"Epoch {e + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, output)
        test_loop(test_dataloader, model, loss_fn, [accuracy, avg_loss], output)
    print("Done!")

    if output["plot"]:
        show_resultes(accuracy, avg_loss)
    return {"acc": accuracy, "loss": avg_loss}


def show_resultes(accuracy, avg_loss):
    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy in %")
    plt.show()

    plt.plot(avg_loss)
    plt.xlabel("Epoch")
    plt.ylabel("AVG Loss")
    plt.show()


def train_loop(dataloader, model, loss_fn, optimizer, output):
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
            if output["cli"]:
                print(f"loss: {loss:>7f}  [{current:>4d}/{size:>4d}]")


def test_loop(dataloader, model, loss_fn, charts, output):
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
    if output["cli"]:

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    charts[0].append((100 * correct))
    charts[1].append(test_loss)


if __name__ == '__main__':
    cls_set1()
