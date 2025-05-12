from sklearn.linear_model import LinearRegression
import joblib
from sklearn import svm
import numpy as np
import os
import xgboost as xgb

# FUNCIONES PARA CARGAR DATASETS PREPROCESADOS
def cargar_datos_train_originales():
    train_data_X = joblib.load('datos_procesados/train/train_data_X.joblib')
    train_data_y = joblib.load('datos_procesados/train/train_data_y.joblib')
    train_data_qids = joblib.load('datos_procesados/train/train_data_qids.joblib')
    return train_data_X, train_data_y, train_data_qids

def cargar_datos_train_pairwise():
    train_data_X = joblib.load('datos_procesados/train_pairwise/train_pairwise_X.joblib')
    train_data_y = joblib.load('datos_procesados/train_pairwise/train_pairwise_y.joblib')
    return train_data_X, train_data_y

# FUNCIONES PARA CREAR MODELOS
def crear_pointwise_model(train_data_X, train_data_y):
    # Entrena un modelo de regresión lineal utilizando los datos de entrenamiento.
    os.makedirs('modelos', exist_ok=True)
    model = LinearRegression()
    model.fit(train_data_X, train_data_y)
    joblib.dump(model, 'modelos/modelo_pointwise.joblib')

def crear_pairwise_model(train_data_X, train_data_y):
    # Entrena un modelo SVM lineal con los datos de entrenamiento (pares)
    os.makedirs('modelos', exist_ok=True)
    model = svm.SVC(kernel='linear')
    model.fit(train_data_X, train_data_y)
    joblib.dump(model, 'modelos/modelo_pairwise.joblib')

def crear_listwise_model(train_data_X, train_data_y, q_group):
    # Entrena un modelo XGBoost utilizando el enfoque Listwise
    os.makedirs('modelos', exist_ok=True)
    # Convertir a formato DMatrix
    dtrain = xgb.DMatrix(train_data_X, label=train_data_y)
    dtrain.set_group(q_group)
    # Parámetros de ranking con NDGC
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg'
    }
    # Entrenamiento del modelo con xgboost
    model = xgb.train(params, dtrain, num_boost_round=100)
    # 6. Guardar el modelo con joblib
    joblib.dump(model, 'modelos/modelo_listwise.joblib')

# FUNCIONES PARA ENFOQUES POINTWISE, PAIRWISE Y LISTWISE
def pointwise():
    # Cargar datos
    train_data_X, train_data_y, _ = cargar_datos_train_originales()
    # Crear y guardar el modelo
    crear_pointwise_model(train_data_X, train_data_y)

def pairwise():
    # Cargar datos
    x_train, y_train = cargar_datos_train_pairwise()
    # Crear y guardar el modelo
    crear_pairwise_model(x_train, y_train)

def listwise():
    #revisar todos los documentos del mismo query
    train_data_X, train_data_y, train_data_qids = cargar_datos_train_originales()
    # Contar la cantidad de documentos por query id, en el orden en que aparecen
    _, indices, counts = np.unique(train_data_qids, return_index=True, return_counts=True)
    q_group = counts[np.argsort(indices)]  # Ordenar para mantener el orden original de aparición
    # Crear y guardar el modelo 
    crear_listwise_model(train_data_X, train_data_y, q_group)

# FUNCIONES PARA ORDENAMIENTO DE DATOS

def main():
    pointwise()
    #pairwise()
    listwise()

if __name__ == '__main__':
    main()