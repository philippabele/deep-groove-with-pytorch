from cls_set1 import cls_set1
from cls_set1 import show_resultes

def main():
    params = {'batch_size': 10,
              'shuffle': True,
              'learning_rate': 1e-6,
              'epochs': 200}
    acc, avg = cls_set1(False, params)
    show_resultes(acc, avg)


if __name__ == '__main__':
    main()
