from utils import load_dataset
import sys
from sklearn.linear_model import LinearRegression
import joblib

def cargar_datasets():
    # Cargar, limpiar y divdidir los datos
    test_data_X, test_data_y, test_data_qids = load_dataset("Datos_Fold_1/test.txt")
    train_data_X, train_data_y, train_data_qids = load_dataset("Datos_Fold_1/train.txt")
    vali_data_X, vali_data_y, vali_data_qids = load_dataset("Datos_Fold_1/vali.txt")

    # Guardar los datos de entrenamiento
    joblib.dump(train_data_X, 'datos_procesados/train/train_data_X.pkl')
    joblib.dump(train_data_y, 'datos_procesados/train/train_data_y.pkl')
    joblib.dump(train_data_qids, 'datos_procesados/train/train_data_qids.pkl')
    # Guardar los datos de prueba
    joblib.dump(test_data_X, 'datos_procesados/test/test_data_X.pkl')
    joblib.dump(test_data_y, 'datos_procesados/test/test_data_y.pkl')
    joblib.dump(test_data_qids, 'datos_procesados/test/test_data_qids.pkl')
    # Guardar los datos de entrenamiento
    joblib.dump(vali_data_X, 'datos_procesados/vali/vali_data_X.pkl')
    joblib.dump(vali_data_y, 'datos_procesados/vali/vali_data_y.pkl')         
    joblib.dump(vali_data_qids, 'datos_procesados/vali/vali_data_qids.pkl')

    return train_data_X, train_data_y

def pointwise_model(train_data_X, train_data_y):
    """
    Entrena un modelo de regresión lineal utilizando los datos de entrenamiento.
    """
    model = LinearRegression()
    model.fit(train_data_X, train_data_y)
    joblib.dump(model, 'modelos/modelo_pointwise.pkl')

def main():
    # Cargar datos
    train_data_X, train_data_y = cargar_datasets()

    # Entrenar modelos
    # Entrenar modelo pointwise
    pointwise_model(train_data_X, train_data_y)

    # Ordenar datos

if __name__ == '__main__':
    main()