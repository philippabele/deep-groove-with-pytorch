from classification_dataset1.cls_set1 import cls_set1
import classification_dataset1.parameter_study as parameter_study

DATA_PATH = "deep_groove_project/dataset/dataset1.csv"

def main():
    cls_set1(dataset = DATA_PATH)
    parameter_study.main(DATA_PATH)
    
    


if __name__ == '__main__':
    main()
