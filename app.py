from flask import Flask, render_template
import pandas as pd
from model import entrenar_modelo, preprocesar_datos
import random

app = Flask(__name__)

# Cargar los datos generados
datos = pd.read_csv('datos_aleatorios.csv')

# Generar una columna de retención inicial con valores balanceados (se desbalanceará con SMOTE)
datos['Retencion'] = [random.choice([1, 0]) for _ in range(len(datos))]

# Entrenar el modelo y preprocesar los datos
modelo, datos = entrenar_modelo(datos)
datos = preprocesar_datos(datos)

# Mapeo para la visualización de datos
sexo_map = {0: 'Masculino', 1: 'Femenino'}
actividades_map = {0: 'No', 1: 'Sí'}
comunicacion_map = {0: 'Ninguno', 1: 'Medio', 2: 'Bueno'}
apoyo_map = {0: 'Regular', 1: 'Bueno'}
tutoria_map = {0: 'No', 1: 'Sí'}
socioeconomico_map = {0: 'Ninguno', 1: 'Medio', 2: 'Bueno'}
retencion_map = {0: 'Abandona', 1: 'No Abandona'}

@app.route('/')
def index():
    estudiantes = datos[['ID', 'Sexo', 'Edad', 'Promedio', 'ActividadesExtracurriculares', 'ComunicacionProfesores',
                         'ApoyoAcademico', 'ParticipacionTutorias', 'NivelSocioeconomico', 'Retencion', 'Prediccion', 'ProbabilidadPeligro']].copy()
    
    # Aplicar los mapeos
    estudiantes['Sexo'] = estudiantes['Sexo'].map(sexo_map)
    estudiantes['ActividadesExtracurriculares'] = estudiantes['ActividadesExtracurriculares'].map(actividades_map)
    estudiantes['ComunicacionProfesores'] = estudiantes['ComunicacionProfesores'].map(comunicacion_map)
    estudiantes['ApoyoAcademico'] = estudiantes['ApoyoAcademico'].map(apoyo_map)
    estudiantes['ParticipacionTutorias'] = estudiantes['ParticipacionTutorias'].map(tutoria_map)
    estudiantes['NivelSocioeconomico'] = estudiantes['NivelSocioeconomico'].map(socioeconomico_map)
    estudiantes['Retencion'] = estudiantes['Retencion'].map(retencion_map)

    return render_template('index.html', estudiantes=estudiantes.to_dict(orient='records'))

@app.route('/detalle/<int:id>')
def detalle(id):
    estudiante = datos[datos['ID'] == id].copy()
    
    # Convertir a texto legible
    estudiante['Sexo'] = estudiante['Sexo'].map(sexo_map)
    estudiante['ActividadesExtracurriculares'] = estudiante['ActividadesExtracurriculares'].map(actividades_map)
    estudiante['ComunicacionProfesores'] = estudiante['ComunicacionProfesores'].map(comunicacion_map)
    estudiante['ApoyoAcademico'] = estudiante['ApoyoAcademico'].map(apoyo_map)
    estudiante['ParticipacionTutorias'] = estudiante['ParticipacionTutorias'].map(tutoria_map)
    estudiante['NivelSocioeconomico'] = estudiante['NivelSocioeconomico'].map(socioeconomico_map)
    estudiante['Retencion'] = estudiante['Retencion'].map(retencion_map)
    
    estudiante = estudiante.to_dict(orient='records')[0]
    return render_template('detalle.html', estudiante=estudiante)

@app.route('/riesgo')
def riesgo():
    # Filtrar estudiantes que no abandonan y cuyo porcentaje de predicción sea >= 0.7
    en_riesgo = datos[(datos['Retencion'] == 1) & (datos['ProbabilidadPeligro'] >= 0.7)][
        ['ID', 'Sexo', 'Edad', 'Promedio', 'Retencion', 'Prediccion', 'ProbabilidadPeligro']
    ].copy()
    
    # Aplicar los mapeos
    en_riesgo['Sexo'] = en_riesgo['Sexo'].map(sexo_map)
    en_riesgo['Retencion'] = en_riesgo['Retencion'].map(retencion_map)
    en_riesgo['Prediccion'] = en_riesgo['Prediccion'].apply(lambda x: f"{x:.2f}")  # Formatear porcentaje de predicción
    en_riesgo['Accion'] = 'Ver detalles'
    
    return render_template('riesgo.html', estudiantes=en_riesgo.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
