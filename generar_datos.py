import random
import csv
import numpy as np  # type: ignore

# Definiciones de valores y probabilidades
sexo_map = {'M': 0, 'F': 1}  # Mapeo de sexo a 0 y 1
sexo_probabilidad = [0.6, 0.4]
actividades = ['Si', 'No']
comunicacion = ['Bueno', 'Medio', 'Ninguno']
apoyo = ['Bueno', 'Regular']
tutoria = ['Si', 'No']
socioeconomico = ['Bueno', 'Medio', 'Ninguno']
retencion = ['Abandona', 'No Abandona']
prob_retencion = [0.3, 0.7]  # Ejemplo de probabilidades

def generar_csv_aleatorio(nombre_archivo, num_filas):
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['ID', 'Sexo', 'Edad', 'Promedio', 'ActividadesExtracurriculares',
                               'ComunicacionProfesores', 'ApoyoAcademico', 'ParticipacionTutorias',
                               'NivelSocioeconomico', 'Retencion'])  # Encabezado

        for i in range(num_filas):
            id = i + 1  # Asigna un ID único para cada fila
            sex = random.choices(list(sexo_map.keys()), weights=sexo_probabilidad, k=1)[0]
            sex_val = sexo_map[sex]  # Convertir a 0 o 1
            age = round(generar_valor_normal(18, 25, True))
            promedio = round(random.uniform(0, 10), 1)
            actividades_val = random.choice(actividades)
            comunicacion_val = random.choice(comunicacion)
            apoyo_val = random.choice(apoyo)
            tutoria_val = random.choice(tutoria)
            socioeconomico_val = random.choice(socioeconomico)
            retencion_val = random.choices(retencion, prob_retencion)[0]
            
            escritor_csv.writerow([id, sex_val, age, promedio, actividades_val, comunicacion_val, apoyo_val,
                                   tutoria_val, socioeconomico_val, retencion_val])

def generar_valor_normal(lim_inferior, lim_superior, forzar_limites=False):
    media = (lim_inferior + lim_superior) / 2
    desviacion_estandar = (lim_superior - lim_inferior) / 6  # 99.7% de los datos estarán dentro de 3 desviaciones estándar

    # Generar número aleatorio distribuido normalmente dentro del rango
    valor = np.random.normal(loc=media, scale=desviacion_estandar)

    if forzar_limites:
        # Ajustar el número aleatorio para asegurarse de que esté dentro del rango
        while valor < lim_inferior or valor > lim_superior:
            valor = np.random.normal(loc=media, scale=desviacion_estandar)

    return valor

# Ejemplo de uso
generar_csv_aleatorio('datos_aleatorios.csv', 1000)  # Genera un archivo con 1000 filas
