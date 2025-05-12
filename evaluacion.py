import joblib

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

def evaluacion_pointwise():
    # Cargar los datos de validación
    vali_data_X, vali_data_y, vali_data_qids = cargar_datos_vali_originales()
    modelo_pointwise = joblib.load('modelos/modelo_pointwise.pkl')

def evaluacion_pairwise():
    # Cargar los datos de validación
    vali_data_X, vali_data_y, vali_data_qids = cargar_datos_vali_pairwise()
    modelo_pairwise = joblib.load('modelos/modelo_pairwise.pkl')

def evaluacion_listwise():
    # Cargar los datos de validación
    vali_data_X, vali_data_y, vali_data_qids = cargar_datos_vali_originales()
    modelo_listwise = joblib.load('modelos/modelo_listwise.pkl')

def main():
    evaluacion_pointwise()
    #evaluacion_pairwise()
    #evaluacion_listwise()

if __name__ == '__main__':
    main()