import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset:

    def __init__(self, dataframe):

        x = dataframe.iloc[1:, 0:2].values
        y = dataframe.iloc[1:, 2].values

        self.sample = torch.tensor(x, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.sample[index, :], self.label[index]
