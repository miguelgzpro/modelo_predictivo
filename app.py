from flask import Flask, render_template  # type: ignore
import pandas as pd  # type: ignore
import random
from model import entrenar_modelo, preprocesar_datos  # type: ignore

app = Flask(__name__)

# Cargar los datos generados
datos = pd.read_csv('datos_aleatorios.csv')

# Probabilidades de las columnas
prob_sexo = [0.5, 0.5]  # Probabilidades para Masculino y Femenino
prob_actividades = [0.6, 0.4]  # Probabilidades para Si y No
prob_comunicacion = [0.5, 0.3, 0.2]  # Probabilidades para Bueno, Medio y Ninguno
prob_apoyo = [0.7, 0.3]  # Probabilidades para Bueno y Regular
prob_tutorias = [0.6, 0.4]  # Probabilidades para Si y No
prob_nivel = [0.4, 0.4, 0.2]  # Probabilidades para Bueno, Medio y Ninguno

# Asignar valores basados en probabilidades
datos['Sexo'] = [random.choices([1, 0], prob_sexo)[0] for _ in range(len(datos))]
datos['ActividadesExtracurriculares'] = [random.choices([1, 0], prob_actividades)[0] for _ in range(len(datos))]
datos['ComunicacionProfesores'] = [random.choices([2, 1, 0], prob_comunicacion)[0] for _ in range(len(datos))]
datos['ApoyoAcademico'] = [random.choices([1, 0], prob_apoyo)[0] for _ in range(len(datos))]
datos['ParticipacionTutorias'] = [random.choices([1, 0], prob_tutorias)[0] for _ in range(len(datos))]
datos['NivelSocioeconomico'] = [random.choices([2, 1, 0], prob_nivel)[0] for _ in range(len(datos))]

# Entrenar el modelo y preprocesar los datos
modelo, datos = entrenar_modelo(datos)
datos = preprocesar_datos(datos)

# Mapeo para la visualización de datos
sexo_map = {1: 'Masculino', 0: 'Femenino'}

@app.route('/')
def index():
    estudiantes = datos[['ID', 'Sexo', 'Edad', 'Promedio', 'Retencion', 'Prediccion']].copy()
    estudiantes['Sexo'] = estudiantes['Sexo'].map(sexo_map)
    return render_template('index.html', estudiantes=estudiantes.to_dict(orient='records'))

@app.route('/detalle/<int:id>')
def detalle(id):
    estudiante = datos[datos['ID'] == id].to_dict(orient='records')[0]
    estudiante['Sexo'] = sexo_map[estudiante['Sexo']]
    return render_template('detalle.html', estudiante=estudiante)

@app.route('/riesgo')
def riesgo():
    en_riesgo = datos[datos['Retencion'] == 'En Peligro'][['ID', 'Sexo', 'Edad', 'Promedio', 'Retencion', 'Prediccion']].copy()
    en_riesgo['Sexo'] = en_riesgo['Sexo'].map(sexo_map)
    en_riesgo['Accion'] = 'Ver detalles'
    return render_template('riesgo.html', estudiantes=en_riesgo.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
