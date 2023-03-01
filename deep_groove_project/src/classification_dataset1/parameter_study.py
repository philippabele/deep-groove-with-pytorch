from cls_set1 import cls_set1
from cls_set1 import show_resultes
import matplotlib.pyplot as plt
import time


def main():
    """ This Function does a Parameter Study with tree different parameters.
        - batch Size
        - learning parameter
        - epochs
    """
    params = {'batch_size': 20,
              'shuffle': True,
              'learning_rate': 1e-7,
              'epochs': 200}

    parameter_batch_size(params, output={"cli": False, "plot": False})
    parameter_learning_rate(params, output={"cli": False, "plot": False})
    parameter_epochs(params, output={"cli": True, "plot": False})


def parameter_epochs(params, output):
    """ parameter_epochs(params, output)
        e.g. params = {
                'batch_size': 20,    # the size of the Batches
                'shuffle': True,            # shuffle the train data
                'learning_rate': 1e-7,      # learning rate of the model
                'epochs': 200}              # number of epochs
            output={
                'cli': False,           # shows console prints when True
                'plot': False}              # shows plots if True

        this Function variates the epochs of the ML models learning Process
        epochs = [10, 50, 100, 200, 500, 1000]
    """
    st = time.time()
    values = []
    epochs = [10, 50, 100, 200, 500, 1000]
    # doing the training of the model for every value in epochs
    for i in range(len(epochs)):
        params["epochs"] = epochs[i]
        values.append(cls_set1(params, output=output))
        values[len(values) - 1]['epochs'] = epochs[i]

    # printing the time it needed
    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 'seconds')

    # generating the Plot for the Accuracy
    for val, i in zip(values, range(len(values))):
        plt.plot(val["acc"], label=val["epochs"])
    plt.legend(loc="lower right")
    plt.title("Accuracy, differing Epochs")
    plt.show()

    # generating the Plot for the average Loss
    for val, i in zip(values, range(len(values))):
        plt.plot(val["loss"], label=val["epochs"])
    plt.legend(loc="upper right")
    plt.title("AVG Loss, differing Epochs")
    plt.show()


def parameter_learning_rate(params, output):
    """ parameter_epochs(params, output)
        e.g. params = {'batch_size': 20,    # the size of the Batches
                'shuffle': True,            # shuffle the train data
                'learning_rate': 1e-7,      # learning rate of the model
                'epochs': 200}              # number of epochs
            output={"cli": False,           # shows console prints when True
                "plot": False}              # shows plots if True

    this Function variates the learning rate of the ML models learning Process
    learning_rates = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    """

    st = time.time()
    values = []
    learning_rates = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # doing the training of the model for every value in learning_rates
    for i in range(len(learning_rates)):
        params["learning_rate"] = learning_rates[i]
        values.append(cls_set1(params, output=output))
        values[len(values) - 1]['learning_rate'] = learning_rates[i]

    # printing the time it needed
    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 'seconds')

    # generating the Plot for the Accuracy
    for val, i in zip(values, range(len(values))):
        plt.plot(val["acc"], label=val["learning_rate"])
    plt.legend(loc="lower right")
    plt.title("Accuracy, differing Learning Rates")
    plt.show()

    # generating the Plot for the Average Loss
    for val, i in zip(values, range(len(values))):
        plt.plot(val["loss"], label=val["learning_rate"])
    plt.legend(loc="upper right")
    plt.title("AVG Loss, differing Learning Rates")
    plt.show()


def parameter_batch_size(params, output):
    """ parameter_epochs(params, output)
        e.g. params = {'batch_size': 20,    # the size of the Batches
                'shuffle': True,            # shuffle the train data
                'learning_rate': 1e-7,      # learning rate of the model
                'epochs': 200}              # number of epochs
            output={"cli": False,           # shows console prints when True
                "plot": False}              # shows plots if True

    this Function variates the batch size of the training and testing data. So the number of data points the model
    gets every iteration.
    learning_rates = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    """
    st = time.time()
    values = []
    batch_sizes = [10, 20, 30, 40, 50, 100]

    # doing the training of the model for every value in batch_size
    for i in range(len(batch_sizes)):
        params["batch_size"] = batch_sizes[i]
        values.append(cls_set1(params, output=output))
        values[len(values)-1]['batch_size'] = batch_sizes[i]

    # printing the time it needed
    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 'seconds')

    # generating the Plot for the Accuracy diagram
    for val, i in zip(values, range(len(values))):
        plt.plot(val["acc"], label=val["batch_size"])
    plt.legend(loc="lower right")
    plt.title("Accuracy, differing Batch Sizes")
    plt.show()

    # generating the Plot for the Average Loss
    for val, i in zip(values, range(len(values))):
        plt.plot(val["loss"], label=val["batch_size"])
    plt.legend(loc="upper right")
    plt.title("AVG Loss, differing Batch Sizes")
    plt.show()


if __name__ == '__main__':
    main()
