import cv2
import face_recognition as fr
import numpy as np
from PIL import Image, ImageDraw
import dlib

foto_control = fr.load_image_file("dia14/FotoA.jpg")
foto_prueba = fr.load_image_file("dia14/FotoC.jpg")

#localizar cara control
lugar_cara_A = fr.face_locations(foto_control)[0]
cara_codificada_A = fr.face_encodings(foto_control)[0]

lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

#mostrar rectanglo

cv2.rectangle(foto_control, (lugar_cara_A[3], lugar_cara_A[0]), (lugar_cara_A[1], lugar_cara_A[2]), (0, 255, 0), 2)

cv2.rectangle(foto_prueba, (lugar_cara_B[3], lugar_cara_B[0]), (lugar_cara_B[1], lugar_cara_B[2]), (0, 255, 0), 2)

#realizar comparacion
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B)


#pasar imagenes a rgb
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#medida de la distancia
distacia = fr.face_distance([cara_codificada_A], cara_codificada_B)

#mostrar resultado 
cv2.putText(foto_prueba,f'{resultado} {distacia.round(2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0 ), 2)

#mostrar imagenes
cv2.imshow('foto_control',foto_control)
cv2.imshow('foto de prueba',foto_prueba)

#mantener el programa abierto
cv2.waitKey(0)
