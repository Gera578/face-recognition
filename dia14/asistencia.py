import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

#crear base de datos
ruta = 'dia14/Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}/{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])
    print(nombres_empleados)


def codificar(imagenes):
    #crear una nueva lista
    lista_codificada = []
    
    #pasar todas las imagenes a rgb
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        #codificar las imagenes
        imagen_codificada = fr.face_encodings(imagen)[0]
        #agregar a la lista
        lista_codificada.append(imagen_codificada)
        
    return lista_codificada


#tomar la asistencia
def asistencia(persona):
    f = open('dia14/registro.csv', 'r+')
    lista_datos = f.readlines()
    nombres_registro = []
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])
    if persona not in nombres_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M:%S')
        f.writelines(f'\n{persona},{string_ahora}')
    else:
        return False
    

lista_empleados_codificada = codificar(mis_imagenes)
print(len(lista_empleados_codificada))


#tomar captura de camara web
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#leer la imagen de la camara
captura.read()

exito, imagen = captura.read()

if not exito:
    print("No se puede abrir la camara")
else:
    #reconocer cara en captura
    cara = fr.face_locations(imagen)
    cara_codificada = fr.face_encodings(imagen, cara)
    #buscar coincidencias
    for coincidencia, ubicacion in zip(cara_codificada, cara ):
        coincidencias = fr.compare_faces(lista_empleados_codificada, coincidencia)
        distancias = fr.face_distance(lista_empleados_codificada, coincidencia)
        
        print(distancias)
        
        indice_de_coincidencia = numpy.argmin(distancias)
        
        #mostrar coincidecias
        if distancias[indice_de_coincidencia] > 0.6:
            print('No hay coincidencias')
        else:
            nombre = nombres_empleados[indice_de_coincidencia]
            print(f'Coincidencia: {nombre}, bienvenido')
            #mostrar la imagen obtenida
            y1, x1, y2, x2 = ubicacion
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imagen, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            asistencia(nombre)
            
            cv2.imshow('imagen web', imagen)
            
            
            #mantener ventana abierta
            cv2.waitKey(0)
