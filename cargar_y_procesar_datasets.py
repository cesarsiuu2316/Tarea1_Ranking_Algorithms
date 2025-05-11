from collections import defaultdict
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

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


def generar_dataset_pairwise(X, y, qids):
    X_pairs = []
    y_pairs = []
    qids_pairs = []  # Para almacenar los qids de los pares de documentos
    query_to_indices = defaultdict(list) # Crea una lista vacía para cada qid agregado
    MAX_DOCS_PER_QUERY = 30

    # Agrupar documentos por query
    for idx, qid in enumerate(qids): # Iterar sobre los índices y qids
        query_to_indices[qid].append(idx) # Almacenar el índice del documento en la lista correspondiente al qid

    for qid, indices in query_to_indices.items(): # Iterar sobre los qids y sus índices
        indices = indices[:MAX_DOCS_PER_QUERY]
        for i in range(len(indices)): # Iterar sobre los índices de los documentos
            for j in range(i + 1, len(indices)): # Comparar cada par de documentos
                idx_i, idx_j = indices[i], indices[j] # Obtener los índices de los documentos, el documento i y todos los documentos j posteriores
                rel_i, rel_j = y[idx_i], y[idx_j] # Obtener los puntajes de relevancia de los documentos

                if rel_i == rel_j: # Si los puntajes de relevancia son iguales, no se agregan porque no aportan información adicional
                    continue

                diff = X[idx_i] - X[idx_j] # Calcular la diferencia entre los vectores de características
                label = 1 if rel_i > rel_j else -1

                # Agregar los pares y sus qids correspondientes
                X_pairs.append(diff)
                y_pairs.append(label)
                qids_pairs.append(qid)

    return np.array(X_pairs), np.array(y_pairs), np.array(qids_pairs)


def cargar_estandarizar_y_guardar_datasets():
    # Crear directorios si no existen
    os.makedirs('datos_procesados/train', exist_ok=True)
    os.makedirs('datos_procesados/test', exist_ok=True)
    os.makedirs('datos_procesados/vali', exist_ok=True)
    os.makedirs('datos_procesados/train_pairwise', exist_ok=True)
    os.makedirs('datos_procesados/test_pairwise', exist_ok=True)
    os.makedirs('datos_procesados/vali_pairwise', exist_ok=True)

    # Cargar, limpiar y divdidir los datos
    test_data_X, test_data_y, test_data_qids = limpiar_dividir_datasets("Fold1/test.txt")
    train_data_X, train_data_y, train_data_qids = limpiar_dividir_datasets("Fold1/train.txt")
    vali_data_X, vali_data_y, vali_data_qids = limpiar_dividir_datasets("Fold1/vali.txt")

    # Estandarizar los datos
    scaler = StandardScaler()
    train_data_X = scaler.fit_transform(train_data_X)
    test_data_X = scaler.transform(test_data_X)
    vali_data_X = scaler.transform(vali_data_X)
    joblib.dump(scaler, 'datos_procesados/scaler.pkl')

    # Guardar los datos de entrenamiento
    joblib.dump(train_data_X, 'datos_procesados/train/train_data_X.pkl')
    joblib.dump(train_data_y, 'datos_procesados/train/train_data_y.pkl')
    joblib.dump(train_data_qids, 'datos_procesados/train/train_data_qids.pkl')
    # Guardar los datos de prueba
    joblib.dump(test_data_X, 'datos_procesados/test/test_data_X.pkl')
    joblib.dump(test_data_y, 'datos_procesados/test/test_data_y.pkl')
    joblib.dump(test_data_qids, 'datos_procesados/test/test_data_qids.pkl')
    # Guardar los datos de validacion
    joblib.dump(vali_data_X, 'datos_procesados/vali/vali_data_X.pkl')
    joblib.dump(vali_data_y, 'datos_procesados/vali/vali_data_y.pkl')         
    joblib.dump(vali_data_qids, 'datos_procesados/vali/vali_data_qids.pkl')

    # 🆕 Generar y guardar datasets Pairwise para cada conjunto
    pairwise_train_X, pairwise_train_y, pairwise_train_qids = generar_dataset_pairwise(train_data_X, train_data_y, train_data_qids)
    pairwise_vali_X, pairwise_vali_y, pairwise_vali_qids = generar_dataset_pairwise(vali_data_X, vali_data_y, vali_data_qids)   
    pairwise_test_X, pairwise_test_y, pairwise_test_qids = generar_dataset_pairwise(test_data_X, test_data_y, test_data_qids)

    # Guardar datasets pairwise
    joblib.dump(pairwise_train_X, 'datos_procesados/train_pairwise/train_pairwise_X.pkl')
    joblib.dump(pairwise_train_y, 'datos_procesados/train_pairwise/train_pairwise_y.pkl')
    joblib.dump(pairwise_train_qids, 'datos_procesados/train_pairwise/train_pairwise_qids.pkl')

    joblib.dump(pairwise_vali_X, 'datos_procesados/vali_pairwise/vali_pairwise_X.pkl')
    joblib.dump(pairwise_vali_y, 'datos_procesados/vali_pairwise/vali_pairwise_y.pkl')
    joblib.dump(pairwise_vali_qids, 'datos_procesados/vali_pairwise/vali_pairwise_qids.pkl')

    joblib.dump(pairwise_test_X, 'datos_procesados/test_pairwise/test_pairwise_X.pkl')
    joblib.dump(pairwise_test_y, 'datos_procesados/test_pairwise/test_pairwise_y.pkl')
    joblib.dump(pairwise_test_qids, 'datos_procesados/test_pairwise/test_pairwise_qids.pkl')


def main():
    cargar_estandarizar_y_guardar_datasets()


if __name__ == '__main__':
    main()