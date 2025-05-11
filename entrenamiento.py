import sys
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

def limpiar_dividir_datasets(path):
    X = [] # Features
    y = [] # relevance labels
    qids = [] # Query IDs
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split() # Delete leading and trailing whitespace and split by whitespace
            y.append(int(parts[0])) # Parse relevance label
            qid = int(parts[1].split(':')[1]) # Parse query ID and get rid of the 'qid:' prefix
            qids.append(qid)
            features = []
            for x in parts[2:]:
                feature = float(x.split(':')[1]) # Parse each feature and get only the value
                features.append(feature) # Append the feature value to the list
            X.append(features) 
    return np.array(X), np.array(y), np.array(qids)

def cargar_guardar_datasets():
    # Cargar, limpiar y divdidir los datos
    test_data_X, test_data_y, test_data_qids = limpiar_dividir_datasets("Datos_Fold_1/test.txt")
    train_data_X, train_data_y, train_data_qids = limpiar_dividir_datasets("Datos_Fold_1/train.txt")
    vali_data_X, vali_data_y, vali_data_qids = limpiar_dividir_datasets("Datos_Fold_1/vali.txt")

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

def pairwise_model(train_data_X, train_data_y):
    #revisar pares de documentos del mismo query
    return

def listwise_model(train_data_X, train_data_y):
    #revisar todos los documentos del mismo query
    return
    

def main():
    return
    # Ordenar datos

if __name__ == '__main__':
    main()