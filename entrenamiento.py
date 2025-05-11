from Tarea1_Ranking_Algorithms.utils import load_dataset
import sys

def main():
    # load and clean dataset
    test_data = load_dataset("Datos_Fold_1/test.txt")
    train_data = load_dataset("Datos_Fold_1/train.txt")
    vali_data = load_dataset("Datos_Fold_1/vali.txt")
    
    


if __name__ == '__main__':
    main()