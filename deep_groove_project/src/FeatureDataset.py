import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):

    def __init__(self, path):
        dataframe = pd.read_csv(path)

        x = dataframe.iloc[:, 2].values
        y = dataframe.iloc[:, 2].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
