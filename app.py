from flask import Flask, render_template # type: ignore
import pandas as pd# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.linear_model import LogisticRegression# type: ignore
from imblearn.over_sampling import SMOTE# type: ignore

app = Flask(__name__)

# Cargar los datos generados
datos = pd.read_csv('datos_aleatorios.csv')

# Preprocesamiento de datos
datos['Sexo'] = datos['Sexo'].map({'M': 1, 'F': 0})
datos['ActividadesExtracurriculares'] = datos['ActividadesExtracurriculares'].map({'Si': 1, 'No': 0})
datos['ComunicacionProfesores'] = datos['ComunicacionProfesores'].map({'Bueno': 2, 'Medio': 1, 'Ninguno': 0})
datos['ApoyoAcademico'] = datos['ApoyoAcademico'].map({'Bueno': 1, 'Regular': 0})
datos['ParticipacionTutorias'] = datos['ParticipacionTutorias'].map({'Si': 1, 'No': 0})
datos['NivelSocioeconomico'] = datos['NivelSocioeconomico'].map({'Bueno': 2, 'Medio': 1, 'Ninguno': 0})

# Separar características y etiqueta
X = datos.drop(['ID', 'Retencion'], axis=1)
y = datos['Retencion']

# Aplicar SMOTE para manejar el desbalanceo de clases
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Entrenar el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
datos['Prediccion'] = modelo.predict(X)

# Función para determinar la razón del riesgo
def razon_de_riesgo(fila):
    razones = []
    if fila['Promedio'] < 6:
        razones.append('Promedio bajo')
    if fila['ComunicacionProfesores'] == 0:
        razones.append('Mala comunicación con profesores')
    if fila['ApoyoAcademico'] == 0:
        razones.append('Falta de apoyo académico')
    if fila['ParticipacionTutorias'] == 0:
        razones.append('No participa en tutorías')
    if fila['NivelSocioeconomico'] == 0:
        razones.append('Nivel socioeconómico bajo')
    return ', '.join(razones) if razones else 'N/A'

# Aplicar la función a los datos
datos['RazonDeRiesgo'] = datos.apply(razon_de_riesgo, axis=1)

@app.route('/')
def index():
    estudiantes = datos.to_dict(orient='records')
    return render_template('index.html', estudiantes=estudiantes)

@app.route('/riesgo')
def riesgo():
    estudiantes_riesgo = datos[datos['Prediccion'] == 0].to_dict(orient='records')
    return render_template('riesgo.html', estudiantes=estudiantes_riesgo)

if __name__ == '__main__':
    app.run(debug=True)
