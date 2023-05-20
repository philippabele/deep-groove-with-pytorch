import pandas as pd
from FeatureDataset import FeatureDataset

from autoPyTorch.api.tabular_classification import TabularClassificationTask



def main():
    #  Loading and preparing Dataframe
    dataframe = pd.read_csv("/dhbw-deep-groove-with-pytorch/deep_groove_project/dataset/dataset1.csv")

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

    api = TabularClassificationTask(
        seed=42,
        n_jobs=4,
        max_models_on_disc = 100
        
    )
    api.search(
        X_train=train_dataset.sample.numpy(),
        y_train=train_dataset.label.numpy(),
        X_test=test_dataset.sample.numpy(),
        y_test=test_dataset.label.numpy(),
        dataset_name='Ball_Bearings',
        optimize_metric='accuracy',
        total_walltime_limit=300,
        func_eval_time_limit_secs=50,
        budget_type ='runtime',
        min_budget = 5,
        max_budget = 1000,
        memory_limit=None
    )

    y_pred = api.predict(test_dataset.sample.numpy())
    score = api.score(y_pred, test_dataset.label.numpy())
    print(score)

    print(api.sprint_statistics())

    api.refit(
        X_train=train_dataset.sample.numpy(),
        y_train=train_dataset.label.numpy(),
        X_test=test_dataset.sample.numpy(),
        y_test=test_dataset.label.numpy(),
        dataset_name="Ball_Bearings",
        # you can change the resampling strategy to
        # for example, CrossValTypes.k_fold_cross_validation
        # to fit k fold models and have a voting classifier
        # resampling_strategy=CrossValTypes.k_fold_cross_validation
    )

    y_pred = api.predict(test_dataset.sample.numpy())
    score = api.score(y_pred, test_dataset.label.numpy())
    print(f"Score: {score}")

    # Print the final ensemble built by AutoPyTorch
    print(api.show_models())


if __name__ == '__main__':
    main()
