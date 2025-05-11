from sklearn.linear_model import LinearRegression
import joblib
from sklearn import svm
import numpy as np
import os

# FUNCIONES PARA CARGAR DATASETS PREPROCESADOS

def cargar_datos_train_originales():
    train_data_X = joblib.load('datos_procesados/train/train_data_X.pkl')
    train_data_y = joblib.load('datos_procesados/train/train_data_y.pkl')
    return train_data_X, train_data_y

def cargar_datos_vali_originales():
    vali_data_X = joblib.load('datos_procesados/vali/vali_data_X.pkl')
    vali_data_y = joblib.load('datos_procesados/vali/vali_data_y.pkl')
    vali_data_qids = joblib.load('datos_procesados/vali/vali_data_qids.pkl')
    return vali_data_X, vali_data_y, vali_data_qids

def cargar_datos_train_pairwise():
    train_data_X = joblib.load('datos_procesados/train_pairwise/train_pairwise_X.pkl')
    train_data_y = joblib.load('datos_procesados/train_pairwise/train_pairwise_y.pkl')
    return train_data_X, train_data_y

def cargar_datos_vali_pairwise():
    vali_data_X = joblib.load('datos_procesados/vali_pairwise/vali_pairwise_X.pkl')
    vali_data_y = joblib.load('datos_procesados/vali_pairwise/vali_pairwise_y.pkl')
    vali_data_qids = joblib.load('datos_procesados/vali_pairwise/vali_pairwise_qids.pkl')
    return vali_data_X, vali_data_y, vali_data_qids

# FUNCIONES PARA CREAR MODELOS

def crear_pointwise_model(train_data_X, train_data_y):
    # Entrena un modelo de regresión lineal utilizando los datos de entrenamiento.
    os.makedirs('modelos', exist_ok=True)
    model = LinearRegression()
    model.fit(train_data_X, train_data_y)
    joblib.dump(model, 'modelos/modelo_pointwise.pkl')

def crear_pairwise_model(train_data_X, train_data_y):
    # Entrena un modelo SVM lineal con los datos de entrenamiento (pares)
    os.makedirs('modelos', exist_ok=True)
    model = svm.SVC(kernel='linear')
    model.fit(train_data_X, train_data_y)
    joblib.dump(model, 'modelos/modelo_pairwise.pkl')
    
def pairwise_model():
    #revisar pares de documentos del mismo query
    x_train, y_train = cargar_datos_train_pairwise()
    crear_pairwise_model(x_train, y_train)

def listwise_model(train_data_X, train_data_y):
    #revisar todos los documentos del mismo query
    return

# FUNCIONES PARA ENFOQUES POINTWISE, PAIRWISE Y LISTWISE


# FUNCIONES PARA ORDENAMIENTO DE DATOS

def main():
    pairwise_model()
    return
    # Ordenar datos

if __name__ == '__main__':
    main()