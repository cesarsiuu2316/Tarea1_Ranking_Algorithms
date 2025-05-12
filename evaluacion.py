from collections import defaultdict
from sklearn.metrics import ndcg_score
import joblib
import numpy as np
import xgboost as xgb

def cargar_datos_vali_originales():
    vali_data_X = joblib.load('datos_procesados/vali/vali_data_X.pkl')
    vali_data_y = joblib.load('datos_procesados/vali/vali_data_y.pkl')
    vali_data_qids = joblib.load('datos_procesados/vali/vali_data_qids.pkl')
    return vali_data_X, vali_data_y, vali_data_qids

def cargar_datos_vali_pairwise():
    vali_data_X = joblib.load('datos_procesados/vali_pairwise/vali_pairwise_X.pkl')
    vali_data_y = joblib.load('datos_procesados/vali_pairwise/vali_pairwise_y.pkl')
    vali_data_qids = joblib.load('datos_procesados/vali_pairwise/vali_pairwise_qids.pkl')
    return vali_data_X, vali_data_y, vali_data_qids

def mean_absolute_precision(y_ordenado, qids):
    # Calcular la precisión media
    query_to_indices = defaultdict(list)
    # Agrupar documentos por query
    for idx, qid in enumerate(qids):
        query_to_indices[qid].append(idx)

    count_queries = 0
    precision_media_absoluta = 0

    # Para cada qid, calcular la precisión media
    for qid, indices in query_to_indices.items():
        count_relevantes = 0
        precision_acumulada = 0
        for idx in range(len(indices)):
            if y_ordenado[indices[idx]] > 0:
                count_relevantes += 1
                precision_acumulada += count_relevantes / (idx + 1)
        if count_relevantes > 0:
            precision_media = precision_acumulada / count_relevantes
            precision_media_absoluta += precision_media
            count_queries += 1  # Solo contamos esta query si tiene al menos un relevante

    if count_queries == 0:
        return 0.0
    precision_media_absoluta /= count_queries
    return precision_media_absoluta

def normalized_discounted_cumulative_gain(vali_data_y, predicciones, qids, k):
    # Agrupar documentos por query (qid)
    query_to_indices = defaultdict(list)
    for idx, qid in enumerate(qids):
        query_to_indices[qid].append(idx)

    ndcg_total = 0
    num_valid_queries = 0

    for qid, indices in query_to_indices.items():
        if len(indices) >= k:
            y_true_q = [vali_data_y[idx] for idx in indices]
            y_pred_q = [predicciones[idx] for idx in indices]

            # sklearn expects shape (1, n_samples) for each list
            ndcg = ndcg_score([y_true_q], [y_pred_q], k=k)
            ndcg_total += ndcg
            num_valid_queries += 1

    return ndcg_total / num_valid_queries    

def ordenar_documentos_por_qid_y_score(vali_data_X, vali_data_y, qids, predicciones):
    # Crear un arreglo para los documentos ordenados por cada qid
    X_ordenado = []
    y_ordenado = []
    qids_ordenado = []
    
    # Para cada qid único (cada consulta)
    query_to_indices = defaultdict(list) # Crea una lista vacía para cada qid agregado

    # Agrupar documentos por query
    for idx, qid in enumerate(qids): # Iterar sobre los índices y qids
        query_to_indices[qid].append(idx) # Almacenar el índice del documento en la lista correspondiente al qid

    for qid, indices in query_to_indices.items(): # Iterar sobre los qids y sus índices
        # Obtener las características, etiquetas y predicciones de los documentos de este qid
        X_qid = vali_data_X[indices]
        y_qid = vali_data_y[indices]
        predicciones_qid = predicciones[indices]
        
        # Ordenar los documentos de este qid por el score/puntaje (de mayor a menor)
        indices_ordenados = np.argsort(predicciones_qid)[::-1]  # Orden descendente
        
        # Añadir los documentos ordenados a las listas finales
        X_ordenado.append(X_qid[indices_ordenados])
        y_ordenado.append(y_qid[indices_ordenados])
        qids_ordenado.append(np.array([qid] * len(indices_ordenados)))  # El qid se repite para todos los documentos
        
    # Convertir las listas en arrays de numpy
    X_ordenado = np.vstack(X_ordenado)  # Unir todos los grupos de documentos
    y_ordenado = np.hstack(y_ordenado)  # Unir todas las etiquetas de relevancia
    qids_ordenado = np.hstack(qids_ordenado)  # Unir todos los qids
    
    return X_ordenado, y_ordenado, qids_ordenado


def evaluacion_pointwise():
    # Cargar los datos de validación
    vali_data_X, vali_data_y, vali_data_qids = cargar_datos_vali_originales()
    modelo_pointwise = joblib.load('modelos/modelo_pointwise.pkl')

    # Realizar predicciones
    predicciones = modelo_pointwise.predict(vali_data_X)
    # Ordenar las predicciones
    X_ordenado, y_ordenado, qids_ordenado = ordenar_documentos_por_qid_y_score(vali_data_X, vali_data_y, vali_data_qids, predicciones)
    # Calcular métricas de evaluación
    mean_absolute_precision_ = mean_absolute_precision(y_ordenado, qids_ordenado)
    print(f"MAP: {mean_absolute_precision_}")
    # NDCG@3
    ndcg = normalized_discounted_cumulative_gain(vali_data_y, predicciones, vali_data_qids, k=5)
    print("NDCG@5:", ndcg)
    # NDCG@5
    ndcg = normalized_discounted_cumulative_gain(vali_data_y, predicciones, vali_data_qids, k=10)
    print("NDCG@10:", ndcg)

def evaluacion_pairwise():
    # Cargar los datos de validación
    vali_data_X, vali_data_y, vali_data_qids = cargar_datos_vali_pairwise()
    modelo_pairwise = joblib.load('modelos/modelo_pairwise.pkl')
    # Realizar predicciones
    predicciones = modelo_pairwise.predict(vali_data_X)
    # Ordenar las predicciones
    X_ordenado, y_ordenado, qids_ordenado = ordenar_documentos_por_qid_y_score(vali_data_X, vali_data_y, vali_data_qids, predicciones)
    # Calcular métricas de evaluación
    mean_absolute_precision_ = mean_absolute_precision(y_ordenado, qids_ordenado)
    print(f"MAP: {mean_absolute_precision_}")

    for i in range(len(vali_data_y)):
        if vali_data_y[i] == -1:
            vali_data_y[i] = 0
        if predicciones[i] == -1:
            predicciones[i] = 0
    # NDCG@3
    ndcg = normalized_discounted_cumulative_gain(vali_data_y, predicciones, vali_data_qids, k=5)
    print("NDCG@5:", ndcg)
    # NDCG@5
    ndcg = normalized_discounted_cumulative_gain(vali_data_y, predicciones, vali_data_qids, k=10)
    print("NDCG@10:", ndcg)

def evaluacion_listwise():
    # Cargar los datos de validación
    vali_data_X, vali_data_y, vali_data_qids = cargar_datos_vali_originales()
    modelo_listwise = joblib.load('modelos/modelo_listwise.pkl')


    predicciones = modelo_listwise.predict(xgb.DMatrix(vali_data_X))
    # Ordenar las predicciones
    X_ordenado, y_ordenado, qids_ordenado = ordenar_documentos_por_qid_y_score(vali_data_X, vali_data_y, vali_data_qids, predicciones)
    # Calcular métricas de evaluación
    mean_absolute_precision_ = mean_absolute_precision(y_ordenado, qids_ordenado)
    print(f"MAP: {mean_absolute_precision_}")
    # NDCG@3
    ndcg = normalized_discounted_cumulative_gain(vali_data_y, predicciones, vali_data_qids, k=5)
    print("NDCG@5:", ndcg)
    # NDCG@5
    ndcg = normalized_discounted_cumulative_gain(vali_data_y, predicciones, vali_data_qids, k=10)
    print("NDCG@10:", ndcg)

def main():
    print("Evaluación de modelos:")
    print("Pointwise:")
    evaluacion_pointwise()
    print("\nPairwise:")
    evaluacion_pairwise()
    print("\nListwise:")    
    evaluacion_listwise()

if __name__ == '__main__':
    main()