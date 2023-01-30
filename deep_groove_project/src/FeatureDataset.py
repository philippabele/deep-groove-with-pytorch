import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset:

    def __init__(self, dataframe):

        x = dataframe.iloc[:, 0:2].values
        y = dataframe.iloc[:, 2].values

        self.sample = torch.tensor(x)
        self.label = torch.tensor(y)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {
            "sample": self.sample[index, :],
            "label": self.label[index]
        }
