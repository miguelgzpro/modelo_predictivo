from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

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
datos['ProbabilidadPeligro'] = modelo.predict_proba(X)[:, 1]

# Aplicar las reglas de retención y predicción
datos['Retencion'] = datos.apply(
    lambda row: 'No está en riesgo' if row['Promedio'] > 7 else 'En Peligro' if row['Retencion'] == 0 else 'Retenido',
    axis=1
)
datos['Prediccion'] = datos.apply(
    lambda row: "-" if row['Retencion'] == 'Retenido' else f"{row['ProbabilidadPeligro'] * 100:.2f}%" if row['Promedio'] <= 7 else "-",
    axis=1
)

# Determinar razón de riesgo
def razon_de_riesgo(fila):
    razones = []
    if fila['Promedio'] <= 7:
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

# Mapeo para la visualización de datos
sexo_map = {1: 'Masculino', 0: 'Femenino'}
actividades_map = {1: 'Sí', 0: 'No'}
comunicacion_map = {2: 'Bueno', 1: 'Medio', 0: 'Ninguno'}
apoyo_map = {1: 'Bueno', 0: 'Regular'}
participacion_map = {1: 'Sí', 0: 'No'}
nivel_map = {2: 'Bueno', 1: 'Medio', 0: 'Ninguno'}

@app.route('/')
def index():
    estudiantes = datos[['ID', 'Sexo', 'Edad', 'Promedio', 'Retencion', 'Prediccion']].copy()
    estudiantes['Sexo'] = estudiantes['Sexo'].map(sexo_map)
    return render_template('index.html', estudiantes=estudiantes.to_dict(orient='records'))

@app.route('/detalle/<int:id>')
def detalle(id):
    estudiante = datos[datos['ID'] == id].to_dict(orient='records')[0]
    estudiante['Sexo'] = sexo_map[estudiante['Sexo']]
    estudiante['ActividadesExtracurriculares'] = actividades_map[estudiante['ActividadesExtracurriculares']]
    estudiante['ComunicacionProfesores'] = comunicacion_map[estudiante['ComunicacionProfesores']]
    estudiante['ApoyoAcademico'] = apoyo_map[estudiante['ApoyoAcademico']]
    estudiante['ParticipacionTutorias'] = participacion_map[estudiante['ParticipacionTutorias']]
    estudiante['NivelSocioeconomico'] = nivel_map[estudiante['NivelSocioeconomico']]
    return render_template('detalle.html', estudiante=estudiante)

@app.route('/riesgo')
def riesgo():
    en_riesgo = datos[datos['Retencion'] == 'En Peligro'][['ID', 'Sexo', 'Edad', 'Promedio', 'Retencion', 'Prediccion']].copy()
    en_riesgo['Sexo'] = en_riesgo['Sexo'].map(sexo_map)
    en_riesgo['Accion'] = 'Ver detalles'
    return render_template('riesgo.html', estudiantes=en_riesgo.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
