import random
import csv
import numpy as np  # type: ignore

sexo = ['M', 'F']
sexo_probabilidad = [0.6, 0.4]

def generar_csv_aleatorio(nombre_archivo, num_filas):
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(['ID', 'Sexo', 'Edad', 'Promedio', 'ActividadesExtracurriculares',
                               'ComunicacionProfesores', 'ApoyoAcademico', 'ParticipacionTutorias',
                               'NivelSocioeconomico'])  # Encabezado sin 'Retencion'

        for i in range(num_filas):
            id = i + 1  # Asigna un ID único para cada fila
            sex = random.choices(sexo, weights=sexo_probabilidad, k=1)[0]  # selección con pesos probabilísticos
            age = round(generar_valor_normal(18, 25, True))
            promedio = round(random.uniform(0, 10), 1)
            actividades = random.choice(['Si', 'No'])
            comunicacion = random.choice(['Bueno', 'Medio', 'Ninguno'])
            apoyo = random.choice(['Bueno', 'Regular'])
            tutoria = random.choice(['Si', 'No'])
            socioeconomico = random.choice(['Bueno', 'Medio', 'Ninguno'])
            
            escritor_csv.writerow([id, sex, age, promedio, actividades, comunicacion, apoyo,
                                   tutoria, socioeconomico])

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
