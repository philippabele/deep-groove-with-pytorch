from cls_set1 import cls_set1
from cls_set1 import show_resultes
import matplotlib.pyplot as plt
import time


def main():

    params = {'batch_size': 20,
              'shuffle': True,
              'learning_rate': 1e-7,
              'epochs': 200}

    parameter_batch_size(params, output={"cli": False, "plot": False})
    parameter_learning_rate(params, output={"cli": False, "plot": False})
    parameter_epochs(params, output={"cli": False, "plot": False})


def parameter_epochs(params, output):
    st = time.time()
    values = []
    epochs = [10, 50, 100, 200, 500, 1000]
    for i in range(len(epochs)):
        params["epochs"] = epochs[i]
        values.append(cls_set1(params, output=output))
        values[len(values) - 1]['epochs'] = epochs[i]

    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 'seconds')

    for val, i in zip(values, range(len(values))):
        plt.plot(val["acc"], label=val["epochs"])
    plt.legend(loc="lower right")
    plt.title("Accuracy, differing Epochs")
    plt.show()

    for val, i in zip(values, range(len(values))):
        plt.plot(val["loss"], label=val["epochs"])
    plt.legend(loc="upper right")
    plt.title("AVG Loss, differing Epochs")
    plt.show()


def parameter_learning_rate(params, output):
    st = time.time()
    values = []
    learning_rates = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    for i in range(len(learning_rates)):
        params["learning_rate"] = learning_rates[i]
        values.append(cls_set1(params, output=output))
        values[len(values) - 1]['learning_rate'] = learning_rates[i]

    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 'seconds')

    for val, i in zip(values, range(len(values))):
        plt.plot(val["acc"], label=val["learning_rate"])
    plt.legend(loc="lower right")
    plt.title("Accuracy, differing Learning Rates")
    plt.show()

    for val, i in zip(values, range(len(values))):
        plt.plot(val["loss"], label=val["learning_rate"])
    plt.legend(loc="upper right")
    plt.title("AVG Loss, differing Learning Rates")
    plt.show()


def parameter_batch_size(params, output):
    st = time.time()
    values = []
    for i in range(10, 101, 10):
        params["batch_size"] = i
        values.append(cls_set1(params, output=output))
        values[len(values)-1]['batch_size'] = i

    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 'seconds')

    for val, i in zip(values, range(len(values))):
        plt.plot(val["acc"], label=val["batch_size"])
    plt.legend(loc="lower right")
    plt.title("Accuracy, differing Batch Sizes")
    plt.show()

    for val, i in zip(values, range(len(values))):
        plt.plot(val["loss"], label=val["batch_size"])
    plt.legend(loc="upper right")
    plt.title("AVG Loss, differing Batch Sizes")
    plt.show()


if __name__ == '__main__':
    main()
