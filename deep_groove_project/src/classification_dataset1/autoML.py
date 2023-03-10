import pandas as pd
from FeatureDataset import FeatureDataset
from FeatureDataset import FeatureDataset


def main():
    #  Loading and preparing Dataframe
    dataframe = pd.read_csv("../../dataset/dataset1.csv")

    dataframe['Label'] = 0
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760, 1, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*2, 2, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*3, 3, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*4, 4, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760*4, 5, inplace=True)

    # shuffle Dataframe
    dataframe = dataframe.sample(frac=1)

    # Creating custom Dataset
    split_index = int(len(dataframe) * 0.75)
    train_dataset = FeatureDataset(dataframe[:split_index])
    test_dataset = FeatureDataset(dataframe[split_index:])

    print(f"length of Datasets - training: {len(train_dataset)}, test: {len(test_dataset)}")


if __name__ == '__main__':
    main()
