from flask import Flask, render_template  # type: ignore
import pandas as pd  # type: ignore
from model import entrenar_modelo, preprocesar_datos  # type: ignore
import random  # Necesario para generar valores aleatorios


app = Flask(__name__)

# Cargar los datos generados
datos = pd.read_csv('datos_aleatorios.csv')

# Generar una columna de retención inicial con valores balanceados (se desbalanceará con SMOTE)
datos['Retencion'] = [random.choice([1, 0]) for _ in range(len(datos))]

# Entrenar el modelo y preprocesar los datos
modelo, datos = entrenar_modelo(datos)
datos = preprocesar_datos(datos)

# Mapeo para la visualización de datos
sexo_map = {'M': 'Masculino', 'F': 'Femenino'}

@app.route('/')
def index():
    estudiantes = datos[['ID', 'Sexo', 'Edad', 'Promedio', 'ActividadesExtracurriculares', 'ComunicacionProfesores',
                         'ApoyoAcademico', 'ParticipacionTutorias', 'NivelSocioeconomico', 'Retencion', 'Prediccion', 'ProbabilidadPeligro']].copy()
    estudiantes['Sexo'] = estudiantes['Sexo'].map(sexo_map)
    return render_template('index.html', estudiantes=estudiantes.to_dict(orient='records'))

@app.route('/detalle/<int:id>')
def detalle(id):
    estudiante = datos[datos['ID'] == id].to_dict(orient='records')[0]
    estudiante['Sexo'] = sexo_map[estudiante['Sexo']]
    return render_template('detalle.html', estudiante=estudiante)

@app.route('/riesgo')
def riesgo():
    en_riesgo = datos[datos['Retencion'] == 'En Peligro'][['ID', 'Sexo', 'Edad', 'Promedio', 'Retencion', 'Prediccion', 'ProbabilidadPeligro']].copy()
    en_riesgo['Sexo'] = en_riesgo['Sexo'].map(sexo_map)
    en_riesgo['Accion'] = 'Ver detalles'
    return render_template('riesgo.html', estudiantes=en_riesgo.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
