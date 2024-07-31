import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from imblearn.over_sampling import SMOTE  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

def entrenar_modelo(datos):
    # Preprocesamiento de datos
    datos_preprocesados = datos.copy()

    # Convertir columnas categóricas a numéricas
    le = LabelEncoder()
    datos_preprocesados['Sexo'] = le.fit_transform(datos_preprocesados['Sexo'])
    datos_preprocesados['ActividadesExtracurriculares'] = le.fit_transform(datos_preprocesados['ActividadesExtracurriculares'])
    datos_preprocesados['ComunicacionProfesores'] = le.fit_transform(datos_preprocesados['ComunicacionProfesores'])
    datos_preprocesados['ApoyoAcademico'] = le.fit_transform(datos_preprocesados['ApoyoAcademico'])
    datos_preprocesados['ParticipacionTutorias'] = le.fit_transform(datos_preprocesados['ParticipacionTutorias'])
    datos_preprocesados['NivelSocioeconomico'] = le.fit_transform(datos_preprocesados['NivelSocioeconomico'])
    
    # Definir variables independientes y dependientes
    X = datos_preprocesados.drop(['ID', 'Retencion'], axis=1)
    y = datos_preprocesados['Retencion']

    # Aplicar SMOTE para manejar el desbalanceo de clases
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    # Entrenar el modelo
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Realizar predicciones
    datos_preprocesados['ProbabilidadPeligro'] = modelo.predict_proba(X)[:, 1]
    datos_preprocesados['Retencion'] = datos_preprocesados['Retencion'].map({1: 'Retenido', 0: 'En Peligro'})
    datos_preprocesados['Prediccion'] = datos_preprocesados.apply(
        lambda row: f"{row['ProbabilidadPeligro'] * 100:.2f}%" if row['Retencion'] == 'En Peligro' else "-",
        axis=1
    )

    return modelo, datos_preprocesados

def razon_de_riesgo(fila):
    razones = []
    if fila['Promedio'] < 6:
        razones.append('El promedio del estudiante es bajo, lo que indica un desempeño académico deficiente.')
    if fila['ComunicacionProfesores'] == 0:
        razones.append('El estudiante no tiene comunicación con los profesores, lo que puede afectar negativamente su rendimiento académico.')
    if fila['ApoyoAcademico'] == 0:
        razones.append('El estudiante no recibe apoyo académico, lo que puede dificultar su progreso en los estudios.')
    if fila['ParticipacionTutorias'] == 0:
        razones.append('El estudiante no participa en tutorías, lo que puede ser una señal de falta de compromiso o de recursos.')
    if fila['NivelSocioeconomico'] == 0:
        razones.append('El estudiante tiene un nivel socioeconómico bajo, lo que puede limitar sus oportunidades de éxito académico.')
    return '<br>'.join(razones) if razones else 'N/A'

def preprocesar_datos(datos):
    datos['RazonDeRiesgo'] = datos.apply(razon_de_riesgo, axis=1)
    return datos
