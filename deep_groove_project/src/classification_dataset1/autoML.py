import pandas as pd

from FeatureDataset import FeatureDataset
import sklearn.datasets
import sklearn.model_selection
from autoPyTorch.api.tabular_classification import TabularClassificationTask


def main():
    #  Loading and preparing Dataframe
    dataframe = pd.read_csv("../../dataset/dataset1.csv")

    dataframe['Label'] = 0
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760, 1, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760 * 2, 2, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760 * 3, 3, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760 * 4, 4, inplace=True)
    dataframe['Label'].where(dataframe['Lifetime'] <= 8760 * 4, 5, inplace=True)

    # shuffle Dataframe
    dataframe = dataframe.sample(frac=1)

    # Creating custom Dataset
    split_index = int(len(dataframe) * 0.75)
    train_dataset = FeatureDataset(dataframe[:split_index])
    test_dataset = FeatureDataset(dataframe[split_index:])

    feature_types = ['Numerical'] * 2 + ['Categorical']

    print(f"{train_dataset.sample.shape}, {train_dataset.label.shape} :: " +
          f"{test_dataset.sample.shape}, {test_dataset.label.shape} :: " +
          f"{feature_types}")

    api = TabularClassificationTask(seed=42)
    api.search(
        X_train=train_dataset.sample,
        y_train=train_dataset.label,
        X_test=test_dataset.sample,
        y_test=test_dataset.label,
        dataset_name='Ball_Bearings',
        optimize_metric='accuracy',
        total_walltime_limit=300,
        func_eval_time_limit_secs=50
    )

    y_pred = api.predict(test_dataset.sample)
    score = api.score(y_pred, test_dataset.label)
    print(score)

    print(api.sprint_statistics())

    api.refit(
        X_train=train_dataset.sample,
        y_train=train_dataset.label,
        X_test=test_dataset.sample,
        y_test=test_dataset.label,
        dataset_name="Ball_Bearings",
        # you can change the resampling strategy to
        # for example, CrossValTypes.k_fold_cross_validation
        # to fit k fold models and have a voting classifier
        # resampling_strategy=CrossValTypes.k_fold_cross_validation
    )

    y_pred = api.predict(test_dataset.sample)
    score = api.score(y_pred, test_dataset.label)
    print(score)

    # Print the final ensemble built by AutoPyTorch
    print(api.show_models())


if __name__ == '__main__':
    main()
