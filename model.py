import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore
from sklearn.preprocessing import LabelEncoder# type: ignore
from imblearn.over_sampling import SMOTE  # type: ignore

def preprocesar_datos(datos):
    # Convertir las columnas categóricas en valores numéricos
    le = LabelEncoder()

    datos['Sexo'] = le.fit_transform(datos['Sexo'])
    datos['ActividadesExtracurriculares'] = le.fit_transform(datos['ActividadesExtracurriculares'])
    datos['ComunicacionProfesores'] = le.fit_transform(datos['ComunicacionProfesores'])
    datos['ApoyoAcademico'] = le.fit_transform(datos['ApoyoAcademico'])
    datos['ParticipacionTutorias'] = le.fit_transform(datos['ParticipacionTutorias'])
    datos['NivelSocioeconomico'] = le.fit_transform(datos['NivelSocioeconomico'])

    return datos

def entrenar_modelo(datos):
    # Preprocesar datos
    datos = preprocesar_datos(datos)

    # Separar características y etiquetas
    X = datos[['Sexo', 'Edad', 'Promedio', 'ActividadesExtracurriculares', 'ComunicacionProfesores', 'ApoyoAcademico',
               'ParticipacionTutorias', 'NivelSocioeconomico']]
    y = datos['Retencion']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Entrenar el modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_res, y_res)

    # Evaluar el modelo (opcional)
    print(f'Precisión en el conjunto de prueba: {modelo.score(X_test, y_test):.2f}')

    # Añadir predicciones y probabilidades al DataFrame original
    datos['Prediccion'] = modelo.predict_proba(X)[:, 1]
    datos['ProbabilidadPeligro'] = modelo.predict_proba(X)[:, 1]

    return modelo, datos
