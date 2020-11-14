#!/usr/bin/env python
# coding: utf-8

# # Tratamiento de imágenes  2020 -   Entregable 1
# # Fecha de entrega: 30/09/2020
# 
# **Importante:** En todos los ejercicios se espera que se entregue comentarios sobre decisiones tomadas en la implementación así como un análisis de los resultados. Estos comentarios y análisis se pueden entregar en secciones agregadas a los notebooks o en un informe aparte.

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import imageio

import cv2
import time

plt.rcParams['image.cmap'] = 'gray'

from matplotlib import collections as mc

import scipy
from scipy import signal
import scipy.fftpack as fft


# In[2]:


## Función auxiliar para imprimir el tiempo de ejecución
def tiempoEjecucion(t_ini):
    print(f'Tiempo de ejecución aproximado: {(time.time()-t_ini):.2f}s')
    
## Función auxiliar para convertir imágenes a escala de grises
def pasarAGris(img):
    img_ = img.copy()
    
    if img_.ndim == 3:
        img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    
    return img_


# ## 1) Líneas de nivel

# Escribir un programa que reciba una imagen y un conjunto de valores y muestre las lineas de nivel correspondientes a dichos valores. Para ello implementar el algoritmo marching squares (Ver por ejemplo: https://en.wikipedia.org/wiki/Marching_squares).
# 
# Aplicarlo a imágenes naturales y artificiales y comparar los resultados con los de la función contour. 
# Comentar los resultados para zonas que tengan cambios suaves de nivel de gris, zonas con bordes bien definidos y zonas con texturas con detalles finos.

# In[3]:


def lineasDeNivel(I, niveles):
    
    #función auxiliar para interpolar 
    def interpolar(lut, celda, nivel):
        ''' 
        celda: celda 2x2 para hacer la interpolación.
        nivel: valor de corte de la curva de con la imagen (lambda).
        lut: array con los lados en los que hay que interpolar el cruce, pueden ser (2,3,4,5) 
              en sentido horario donde 2 es el lado superior. En caso de ser (0, 1) mantiene su valor.
              
        Devuelve un segmento de la forma [(x_salida, y_salida), (x_llegada, y_llegada)]
        normalizado entre [0, 1]. 
        '''
        segmento = np.zeros((2,2))
        
        for i in range(2):
            for j in range(2):
                 
                if lut[i,j] == 1:
                    segmento[i,j] = 1
                elif lut[i,j] == 2:
                    segmento[i,j] = (nivel - celda[0,0])/(celda[0,1] - celda[0,0])
                elif lut[i,j] == 3:
                    segmento[i,j] = (nivel - celda[0,1])/(celda[1,1] - celda[0,1])
                elif lut[i,j] == 4:
                    segmento[i,j] = (nivel - celda[1,0])/(celda[1,1] - celda[1,0])
                elif lut[i,j] == 5:
                    segmento[i,j] = (nivel - celda[0,0])/(celda[1,0] - celda[0,0])
        
        return segmento
    
    # LUT con codificación del segmento a interpolar

    LUT = np.zeros((16,2,2))  
    LUT[0,:,:] = [[0,0],[0,0]]
    LUT[1,:,:] = [[0, 5],[4, 1]]
    LUT[2,:,:] = [[4, 1],[1, 3]]
    LUT[3,:,:] = [[0, 5],[1, 3]]
    LUT[4,:,:] = [[2, 0],[1, 3]]
    LUT[5,:,:] = [[0,0],[0,0]]
    LUT[6,:,:] = [[4, 0],[2, 1]]
    LUT[7,:,:] = [[0, 5],[2, 0]]
    LUT[8,:,:] = [[0, 5],[2, 0]]
    LUT[9,:,:] = [[4, 0],[2, 1]]
    LUT[10,:,:] = [[0,0],[0,0]]
    LUT[11,:,:] = [[2, 0],[1, 3]]
    LUT[12,:,:] = [[0, 5],[1, 3]]
    LUT[13,:,:] = [[4, 1],[1, 3]]
    LUT[14,:,:] = [[0, 5],[4, 1]]
    LUT[15,:,:] = [[0,0],[0,0]]
    # -------------------------------
    
    M, N = I.shape
    
    # array de arrays de segmentos por nivel
    lineasDeNivel = []
    
    for nivel in niveles:
        # almaceno los segmentos para cada nivel
        segmentosNivel = []
        
        # Devuelve un 1 en todos los pixeles cuyo valor supere el del nivel (lambda)
        J = np.floor(I >= nivel).astype(int)

        for i in range(M-1):
            for j in range(N-1):
                # celda original con la que se trabaja
                cel = I[i:i+2, j:j+2].astype(int)
                
                # paso de binario a decimal y guardo el tipo de clase
                clase = 2**3*J[i,j] + 2**2*J[i,j+1] + 2*J[i+1,j+1] + J[i+1,j]
                
                # Se deja la abscisa primero y la ordenada despues para el ploteo
                vertice = np.array([j, i])
    
                # Distingo del caso ambiguo de las clases 5 y 10
                if clase == 5:
                    valorCentro = np.mean(I[i:i+2,j:j+2]) >= nivel
                    if valorCentro:
                        segmento1 = vertice + interpolar(lut=LUT[4], celda=cel, nivel=nivel)
                        segmento2 = vertice + interpolar(lut=LUT[14], celda=cel, nivel=nivel)
                        segmentosNivel.append(segmento1)
                        segmentosNivel.append(segmento2)
                    else:
                        segmento1 = vertice + interpolar(lut=LUT[2], celda=cel, nivel=nivel)
                        segmento2 = vertice + interpolar(lut=LUT[8], celda=cel, nivel=nivel)
                        segmentosNivel.append(segmento1)
                        segmentosNivel.append(segmento2)
                        # end if clase == 5
                elif clase == 10:
                    valorCentro = np.mean(I[i:i+2,j:j+2]) >= nivel
                    if valorCentro:
                        segmento1 = vertice + interpolar(lut=LUT[2], celda=cel, nivel=nivel)
                        segmento2 = vertice + interpolar(lut=LUT[8], celda=cel, nivel=nivel)
                        segmentosNivel.append(segmento1)
                        segmentosNivel.append(segmento2)
                    else:
                        segmento1 = vertice + interpolar(lut=LUT[4], celda=cel, nivel=nivel)
                        segmento2 = vertice + interpolar(lut=LUT[14], celda=cel, nivel=nivel)
                        segmentosNivel.append(segmento1)
                        segmentosNivel.append(segmento2)
                        # end if clase == 10
                elif clase != 0 and clase != 15:
                    segmento = vertice + interpolar(lut=LUT[clase], celda=cel, nivel=nivel)
                    segmentosNivel.append(segmento)
                
        # Agrego los segmentos al array principal
        segmentosNivel = np.array(segmentosNivel)
        lineasDeNivel.append(segmentosNivel)
    
    return np.array(lineasDeNivel, dtype='object') 


# In[4]:


files = ['./imagenes/degrade.jpg', './imagenes/5.1.12.tiff']

img1 = pasarAGris(imageio.imread(files[0]))
img2 = pasarAGris(imageio.imread(files[1]))


# In[5]:


t_ini = time.time()

# Hallar lineas de nivel
niveles = [30, 60, 100, 150, 200]
linNiv_img1 = lineasDeNivel(img1, niveles)
linNiv_img2 = lineasDeNivel(img2, niveles)

tiempoEjecucion(t_ini)


# In[6]:


t_ini = time.time()

#### UTILIZANDO CONTOUR DE OPENCV

# Pasar imágenes a BGR para poder poner color sobre ellas
img1_contour = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_contour = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# Generar colores aleatorios para opencv
col_cv2 = np.random.randint(0, 255, size=(len(niveles), 3))
col_cv2 = [[int(col_cv2[x][y]) for y in range(3)] for x in range(len(niveles))] 

# Hallar contornos y dibujarlos para cada nivel
for i in range(len(niveles)):
    ret, thresh = cv2.threshold(img1, niveles[i], 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1_contour, contours, -1, color=tuple(col_cv2[i]), thickness=1)

    ret, thresh = cv2.threshold(img2, niveles[i], 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2_contour, contours, -1, color=tuple(col_cv2[i]), thickness=1)


tiempoEjecucion(t_ini)


# In[7]:


t_ini = time.time()

colores = ['lime', 'r', 'g', 'b', 'orange', 'yellow', 'm', 'cyan']

plt.figure(figsize=(12,12))

# img1 
ax1 = plt.subplot(2,2,1)
ax1.imshow(img1, cmap='gray', vmin=0, vmax=255)
for lineas, col in zip(linNiv_img1,colores):
    lc = mc.LineCollection(lineas, colors=col)
    ax1.add_collection(lc)
ax1.set_title(f'Curvas de nivel para valores {niveles}')
plt.axis('off')

# img1 cv2 contour
ax2 = plt.subplot(2,2,2)
ax2.imshow(img1_contour, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Función Contour de OpenCV')
plt.axis('off')

# img2
ax3 = plt.subplot(2,2,3)
ax3.imshow(img2, cmap='gray', vmin=0, vmax=255)
for lineas, col in zip(linNiv_img2,colores):
    lc = mc.LineCollection(lineas, colors=col)
    ax3.add_collection(lc)
ax3.set_title(f'Curvas de nivel para valores {niveles}')
plt.axis('off')

# img1 cv2 contour
ax4 = plt.subplot(2,2,4)
ax4.imshow(img2_contour, cmap='gray', vmin=0, vmax=255)
ax4.set_title('Función Contour de OpenCV')
plt.axis('off')

plt.tight_layout()
plt.show()

tiempoEjecucion(t_ini)


# #### Comentarios
# Se puede observar que cuando la imagen tiene bordes bien definidos los algoritmos arrojan líneas bastante uniformes, como si fuera un trazo "natural". Sin embargo, cuando el algoritmo se enfrenta a situaciones de bordes poco definidos o degrades, las trazas son irregulares. Estos fenómenos se pueden apreciar en la imagen del reloj, la estructura del mismo esta bastante bien definida y por lo tanto el trazo es uniforme, mientras que en las regiones de la reflexión de la foto o el libro el trazo se vuelve zigzageante. El caso más extremo de un trazo poco uniforme se da en el borde superior.
# 
# Hay una clara diferencia entre el resultado de la función implementada y la de opencv, esta última pareciera que es más sensible a variaciones y las lineas quedan más ruidosas; o quizás por la manera que implementa el dibujado de las lineas sobre la imagen hace que así parezca. 

# ## 2) Histograma
# 
# Imágenes sugeridas: parot1.png, parot2.png, strike1.png, strike2.png

# **a)** Implementar una función que calcule el histograma e histograma acumulado de una imagen. Mostrar resultados para algunas imágenes.

# In[8]:


def histograma(I, nBins):
    histograma = np.zeros(nBins)
    umbral = 256/nBins
    
    M, N = I.shape

    for i in np.arange(M):
        for j in np.arange(N):
            bin_corresp = int(I[i,j] // umbral)
            histograma[bin_corresp] += 1
            
    return histograma / (M*N)


def histograma_acumulado(hist):
    hist_acc = np.zeros(hist.size)
    for i in np.arange(hist_acc.size):
        hist_acc[i] = hist_acc[i-1] + hist[i]
        
    return hist_acc


# In[9]:


t_ini = time.time()

nBins = 256
ancho = 256/nBins
intervalos = np.arange(0, 256, ancho)

######## IMAGEN UNO

file = './imagenes/kitten.jpg'
img1 = pasarAGris(imageio.imread(file))

# Calcular histograma y acumulado
hist_img1 = histograma(img1, nBins)
hist_acc_img1 = histograma_acumulado(hist_img1)

# Mostrar resultados
plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen {file}')
plt.axis('off')

plt.subplot(1,3,2)
plt.bar(intervalos, hist_img1, width=ancho, align='edge')
plt.title(f'Histograma {file}')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Frecuencia de ocurrencia normalizada')

plt.subplot(1,3,3)
plt.plot(np.arange(256), hist_acc_img1)
plt.title(f'Histograma acumulado {file}')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Cantidad acumulada normalizada')
plt.show()


######## IMAGEN DOS

file = './imagenes/strike1.png'
img2 = pasarAGris(imageio.imread(file))

# Calcular histograma y acumulado
hist_img2 = histograma(img2, nBins)
hist_acc_img2 = histograma_acumulado(hist_img2)

# Mostrar resultados
plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen {file}')
plt.axis('off')

plt.subplot(1,3,2)
plt.bar(intervalos, hist_img2, width=ancho, align='edge')
plt.title(f'Histograma para figura {file}')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Frecuencia de ocurrencia normalizada')

plt.subplot(1,3,3)
plt.plot(np.arange(256), hist_acc_img2)
plt.title(f'Histograma acumulado {file}')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Cantidad acumulada normalizada')
plt.show()

tiempoEjecucion(t_ini)


# **b)** Implementar la ecualización de histogramas. Mostrar la imagen e histograma antes y después de la ecualización.

# In[10]:


def ecualizarHistograma(I):
    
    I_ecualizada = I.copy()
    
    hist_acc = histograma_acumulado(histograma(I, 256)) * 255
    
    def evaluarPixel(p):
        return int(np.round(hist_acc[p]))
    
    for i in np.arange(256):
        I_ecualizada[I==i] = evaluarPixel(i)

    return I_ecualizada


# In[11]:


t_ini = time.time()

file = './imagenes/strike1.png'
I = pasarAGris(imageio.imread(file))


nBins = 256
ancho = 256/nBins
intervalos = np.arange(0, 256, ancho)

# Calcular histograma y acumulado
hist_I = histograma(I, nBins)
hist_acc_I = histograma_acumulado(hist_I)

# ECUALIZAR IMAGEN
I_ecualizada = ecualizarHistograma(I)
hist_I_ecu = histograma(I_ecualizada, nBins)
hist_acc_I_ecu = histograma_acumulado(hist_I_ecu)


### Plotear resultados
plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.bar(intervalos, hist_I, width=ancho, align='edge')
plt.title(f'Histograma para figura {file}')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Frecuencia de ocurrencia normalizada')

plt.subplot(1,2,2)
plt.plot(np.arange(256), hist_acc_I)
plt.title(f'Histograma acumulado para {file}')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Cantidad acumulada normalizada')
plt.show()


plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.bar(intervalos, hist_I_ecu, width=ancho, align='edge')
plt.title(f'Histograma para figura {file} ecualizada')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Frecuencia de ocurrencia normalizada')

plt.subplot(1,2,2)
plt.plot(np.arange(256), hist_acc_I_ecu)
plt.title(f'Histograma acumulado para {file} ecualizada')
plt.xlabel('Valor de pixel en escala de grises')
plt.ylabel('Cantidad acumulada normalizada')
plt.show()


plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.imshow(I, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(I_ecualizada, cmap='gray',  vmin=0, vmax=255)
plt.title('Imagen ecualizada')
plt.axis('off')
plt.tight_layout()
plt.show()

tiempoEjecucion(t_ini)


# #### Comentarios
# Se puede observar el resultado de ecualizar el histograma bien claro en la bandera de la izquierda, en la imagen original prácticamente es ilegible una sección de la bandera, mientras que en la imagen ecualizada es posible distinguir sin problemas el texto de la misma.
# 
# Luego de realizar una ecualización muchas veces queda en evidencia cómo hay información "escondida" que a simple vista por el rango dinámico que se maneja no es posible distinguir. Podría ser útil un pre-procesamiento de este tipo para combinarlo con la parte anterior a la hora de hacer detección de bordes.
# 

# 
# 
# 
# 
# **c)** Implementar una función que realice el matching de histogramas entre un par de imágenes. 
# 
# (Ver por ejemplo: http://paulbourke.net/miscellaneous/equalisation/ y https://en.wikipedia.org/wiki/Histogram_matching )
# 
# (Artículo interesante que hace algo levemente diferente a lo que se pide en este ejercicio: https://www.ipol.im/pub/art/2016/140/)
# 
# 

# In[12]:


## Devuelve una imagen, la cual corresponde a ajustar el histograma de img al histograma de imgBase
def matching(I, imgBase):
 
    M, N = I.shape
    imgMatch = np.zeros((M, N))

    # distribución acumulada para referencia y salida
    F_ref = histograma_acumulado(histograma(imgBase, 256)) 
    F_target = histograma_acumulado(histograma(I, 256))

    for i in range(M):
        for j in range(N):
            # original value
            value_in = I[i,j]
            # matching value
            target = F_target[value_in]
            # find out value 
            diff = np.abs(F_ref - target)
            value_out = np.where(diff == np.min(diff))[0][0]

            imgMatch[i,j] = int(value_out)

    return imgMatch


# **d)** Aplique el matching de histogramas entre dos imágenes (por ejemplo entre parot1.png y parot2.png y/o entre strike1.png y strike2.png). 
# 
# Muestre y analice los resultados.

# In[13]:


t_ini = time.time()

im_ref = pasarAGris(imageio.imread('./imagenes/parot2.png'))
im = pasarAGris(imageio.imread('./imagenes/parot1.png'))

im_matched = matching(im, im_ref)

# Mostrar resultados
## Imágenes
plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt.imshow(im_ref, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de referencia')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(im, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(im_matched, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen matcheada')
plt.axis('off')

plt.tight_layout()
plt.show()

## Histogramas
intervalos = np.arange(0, 256)

plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.bar(intervalos, histograma(im_ref, 256), width=1, align='edge')
plt.title('Histograma referencia')

plt.subplot(1,3,2)
plt.bar(intervalos, histograma(im, 256), width=1, align='edge')
plt.title('Histograma original')

plt.subplot(1,3,3)
plt.bar(intervalos, histograma(im_matched, 256), width=1, align='edge')
plt.title('Histograma matcheado')

plt.tight_layout()
plt.show()

tiempoEjecucion(t_ini)


# El resultado obtenido es bueno, de todas formas es notorio como en ciertas regiones con texturas y sombras delicadas, como en los laterales de las ramas más gruesas del árbol la imagen matcheada pierde algo de detalle. Una posible solución para esto es aplicar una idea como la del paper citado en el encabezado de este ejercicio, que trata de generar una especie de compromiso entre la distribución de la imagen original y la de referencia, obteniendo resultados bastante buenos similares a lo que haría la funcion de HDR en una cámara fotográfica.

# ## 3) Transformaciones geométricas

# **a)** Implementar las interpolaciones vecino más cercano y bilineal.

# In[14]:


# Devuelve el nivel de gris interpolado correspondiente al punto
# punto = (fila,columna)  
# Si el punto está fuera de la imagen devuelve el valor "color_de_fondo" que por defecto es negro
def bilineal(I, punto, color_de_fondo):
    
    M, N = I.shape
    
    valor_interpolado = color_de_fondo
    
    i = punto[0]
    j = punto[1]
    
    if j >= 0 and j < N-1 and i >= 0 and i < M-1:
        i_f = np.floor(i).astype(int)
        j_f = np.floor(j).astype(int)
        i_c = np.floor(i+1).astype(int)
        j_c = np.floor(j+1).astype(int)
        
        valor_interpolado = (i_c - i)*(j_c - j)*I[i_f, j_f] + (i_c - i)*(j - j_f)*I[i_f, j_c] +                             (i - i_f)*(j - j_f)*I[i_c, j_c] + (i - i_f)*(j_c - j)*I[i_c, j_f]
    
    return valor_interpolado


# Devuelve el nivel de gris interpolado correspondiente al punto
# punto = (fila,columna)  
# Si el punto está fuera de la imagen devuelve el valor "color_de_fondo" que por defecto es negro
def vecino(img, punto, color_de_fondo=0):
    
    M, N = img.shape[0:2]
    
    valor_interpolado = color_de_fondo
    
    fila = punto[0]
    columna = punto[1]
    
    if columna >= 0 and columna < N-0.5 and fila >= 0 and fila < M-0.5:
        i = np.round(fila).astype(int)
        j = np.round(columna).astype(int)
        valor_interpolado = img[i,j]
    
    return valor_interpolado


# **b)** Implementar la funcion transformar, que dada una imagen y una matriz realice la transformacion correspondiente. La transformación deberá poder realizarse utilizando interpolación de vecino más cercano o bilineal.

# In[15]:


# Devuelve la imagen transformada a partir de la matriz T
# Se debe poder elegir entre interpolacion vecino y bilineal 
# Si el punto está fuera de la imagen devuelve el valor "color_de_fondo" que por defecto es negro
def transformar(img, T, tipo_interpolacion='vecino', color_de_fondo=0, extender_V=False, extender_H=False):
    
    inv_T = np.linalg.inv(T)
    I_salida = np.ones_like(img) * color_de_fondo
    
    if extender_V:
        I_salida = np.vstack((img,np.zeros_like(img)))
    if extender_H:
        I_salida = np.hstack((img,np.zeros_like(img)))
        
    M, N = I_salida.shape[0:2]
        
    error_infinito = False
    
    for i_d in range(M):
        for j_d in range(N):
            punto = inv_T @ [i_d, j_d, 1]
            if abs(punto[2]) > np.finfo(float).eps:
                punto = punto/punto[2]
            else:
                punto = [0, 0, 1]
                error_infinito = True
            
            if tipo_interpolacion == 'vecino':
                I_salida[i_d, j_d] = vecino(img, punto, color_de_fondo)
            elif tipo_interpolacion == 'bilineal':
                I_salida[i_d, j_d] = bilineal(img, punto, color_de_fondo)
            else:
                raise ValueError('El tipo de interpolación debe ser "vecino" o "bilineal".') 
                
    if error_infinito:
        print('Atención: Algunos puntos están mapeados al infinito')
        
    return I_salida


# ### Transformaciones isométricas, afines y proyectivas  

# **c)** Mostrar la matriz de transformación junto con el la imagen obtenida para las siguientes interpolaciones:
# 
#   **1-** Mostrar una imagen simulando que además ésta se refleja sobre una superficie plana horizontal. Modificar el brillo y contraste de la imagen reflejada para hacer la reflexión más realista. Indicar qué tipo de proyección se realizó para simular la reflexión.
#   
# 

# In[16]:


def reflejarImagen(img, angulo=60, stretch=2, brillo=140):
    '''Recibe una imagen y devuelve una copia con su reflejo.
    
    Parámetros:
    
   angulo: [0, pi] Ángulo de proyección en sentido antihorario con respecto al eje Y.
   stretch: Factor de reducción/ampliación de la sombra en su eje vertical.
   brillo: [0, 255] Valor de la constante a utilizar en la función de contraste aplicada al reflejo.'''
    
    def modificarBrillo(X, brillo):
        return (X*((255-brillo)/255) + brillo).astype(int)
    
    
    if angulo == 0:
        stretch = 1
        
    # Matrices de transformación a utilizar
    ang = angulo * np.pi / 180
    
    T_tangente = np.array([[1, 0, 0],
                      [np.tan(ang), 1, 0],
                      [0, 0, 1]])
    
    T_stretch_y = np.array([[1/stretch, 0 , 0], 
                        [0, 1, 0], 
                        [0, 0, 1]])
    
    T_reflejo = T_tangente @ T_stretch_y
    
    ### CALCULO DE HASTA DONDE EXTENDER LA IMAGEN PARA EL REFLEJO
    
    M, N = img.shape
    
    # Aplicando trigonometría
    
    M_ext = (M/stretch)*np.sqrt(1 + np.tan(ang)**2)* np.cos(ang)
    M_ext = int(np.ceil(M_ext))
      
    N_ext = (M/stretch)*np.sqrt(1 + np.tan(ang)**2)* np.sin(ang)
    N_ext = int(np.ceil(N_ext))
    
    ##############################################################
    
    # Generar la imagen reflejo
    img_espejada = np.flipud(img)
    img_reflejo = transformar(img_espejada, T_reflejo, 'bilineal', color_de_fondo=255, extender_H=True)
    
    M_ref, N_ref = img_reflejo.shape
    
    # Extender imagen original
    if (N+N_ext) > N_ref:
        img = np.hstack((img, np.ones((M, N_ref-N))*255))
    else:
        img = np.hstack((img, np.ones((M, N_ext))*255))
    
    
    # Recortar imagen reflejada
    img_reflejo = img_reflejo[:M_ext, :N+N_ext]
    
    img_reflejo = modificarBrillo(img_reflejo, brillo)
    
    # Acoplar imagen y reflejo
    img_salida = np.vstack((img, img_reflejo))

    return img_salida, T_reflejo


# In[17]:


t_ini = time.time()

I = pasarAGris(imageio.imread('./imagenes/landscape.jpg'))

# Reflejar imagen
I_reflejo, T_reflejo = reflejarImagen(I, angulo=60, stretch=4, brillo=120)

# Mostrar resultados
plt.figure(figsize=(8,6))
plt.imshow(I, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')
plt.show()

plt.figure(figsize=(12,9))
plt.imshow(I_reflejo, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen con reflejo')
plt.axis('off')
plt.show()

# Imprimir matriz de transformación
print()
print('Matriz de transformación para el reflejo')
print(T_reflejo)
print()

tiempoEjecucion(t_ini)


# Para generar la imagen con su reflejo se plantearon dos pasos diferentes, el primero fue generar el reflejo de la imagen deseada y en segundo lugar acoplar el reflejo con dicha imagen.
# 
# Para realizar el reflejo se utilizó una matriz de cizallamiento con un ángulo variable entre [0, $\pi$] medido en sentido antihorario desde el eje Y, seguido de una contracción en la dirección vertical del cizallamiento que pueda contraer o expandir el reflejo. Luego, se aplico una función de contraste de la forma
# 
# $$v_o = v \left( \frac{255 - cte}{255} \right) + cte$$ 
# 
# donde la variable $cte$ es el parámetro que se le puede pasar a la función, además del ángulo anteriormente mencionado.
# 
# Por último, para acoplar las dos imágenes se descartan las secciones excedentes en el lienzo (secciones en blanco) del reflejo y se hace un matcheo de dimensiones para que los dos bordes coincidan y no se produzca un desfasaje.

#   **2-** Mostrar una imagen simulando que ésta está dibujada sobre un muro vertical que se observa en prespectiva. Indicar qué tipo de proyección se realizó para simular la reflexión.
# 

# In[18]:


t_ini = time.time()

I = pasarAGris(imageio.imread('./imagenes/landscape.jpg'))

ang = 10 * np.pi / 180

# Matrices de transformación
T_stretch_x = np.array([[1, 0, 0], 
                    [0, 1.5, 0],
                    [0, 0, 1]])

T_perspectiva = np.array([[1, np.tan(ang), 0], 
                    [0, 1, 0],
                    [0, 0.0005, 1]])

# Componer transformaciones
T_muro = T_perspectiva @ T_stretch_x

# Hallar imagen transformada
I_muro = transformar(I, T_muro, 'bilineal', color_de_fondo=255)

# Mostrar resultados
plt.figure(figsize=(12,8))
plt.imshow(I, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')
plt.show()

plt.figure(figsize=(12,8))
plt.imshow(I_muro, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen en perspectiva')
plt.axis('off')
plt.show()

tiempoEjecucion(t_ini)


# Para realizar esta transformación se aplico algo similar a la parte anterior, utilizando un doble cizallamiento pero esta vez en las direcciones verticales, lo que produce una "inclinación" de la imagen y la sensación de que se esta viendo en perspectiva desde un costado.

# ### Rotación y composición de matrices
# **d)** Mostrar ejemplos de rotaciones con origen en el vértice superior izquierdo de la imagen (origen de coordenadas)

# In[19]:


t_ini = time.time()

img = pasarAGris(imageio.imread('./imagenes/trapo.jpg'))

alto, ancho = img.shape

interpolacion = 'vecino'

# Array de ángulos para rotar
angulos = [30*np.pi/180, 10*np.pi/180, 70*np.pi/180]

#### Mostrar resultados
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Imagen original')
plt.axis('off')
    
for angulo, subp in zip(angulos, range(2,5)):
    T_rot_orig = np.array([[np.cos(angulo), np.sin(angulo), 0], 
                           [-np.sin(angulo), np.cos(angulo), 0], 
                           [0, 0, 1]])

    I_destino = transformar(img, T_rot_orig, tipo_interpolacion='bilineal', color_de_fondo=0)

    plt.subplot(2,2,subp)
    plt.imshow(I_destino)
    plt.title(f'Imagen rotada {np.round(angulo*180/np.pi)}°')
    plt.axis('off')
plt.tight_layout()
plt.show()
    
tiempoEjecucion(t_ini)


# **e)** Mostrar una rotación con origen en el centro de la imagen.
# 
# La transformación de rotación en el centro de la imagen puede calcularse como la composición de una traslación, una rotación y una traslación.  

# In[20]:


t_ini = time.time()

file = './imagenes/trapo.jpg'
img = pasarAGris(imageio.imread(file))

alto, ancho = img.shape

### Muestre las matrices correspondientes a las tres transformaciones

# Ángulo de la rotación en radianes
angulo = 30 * np.pi / 180

# Matriz de rotación con respecto al origen
T_rot_orig = np.array([[np.cos(angulo), np.sin(angulo), 0], 
                       [-np.sin(angulo), np.cos(angulo), 0], 
                       [0, 0, 1]])

# Coordenadas del centro de la imagen
i_trasl = alto/2
j_trasl = ancho/2

# Matriz de traslación hasta el origen
T_trasl_orig = np.array([[1, 0, -i_trasl],
                        [0, 1, -j_trasl], 
                        [0, 0, 1]])

# Matriz de traslación hasta el centro de la imagen
T_trasl_centro = np.array([[1, 0 , i_trasl], 
                          [0, 1, j_trasl], 
                          [0, 0, 1]])

# Calcule la matriz de rotación en el centro como la composición de las tres transformaciones anteriores
T_rot_centro = T_trasl_centro @ T_rot_orig @ T_trasl_orig


# Aplique la transformación y muestre el resultado
I_destino = transformar(img, T_rot_centro, tipo_interpolacion='bilineal', color_de_fondo=0)

# Mostrar resultados
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img)
plt.title('Imagen original')
plt.scatter(ancho/2, alto/2,c='r', marker='+')
plt.axis('off')

plt.subplot(122)
plt.imshow(I_destino)
plt.title(f'Imagen rotada {np.round(angulo*180/np.pi)} °')
plt.scatter(ancho/2, alto/2,c='r', marker='+')
plt.axis('off')
plt.tight_layout()
plt.show()

tiempoEjecucion(t_ini)


# ## Mostrar la degradación que sufre una imagen al realizar los distintos tipos de interpolaciones:  
# Rotar una imagen alrededor del centro de la imagen 36 veces de a 10º cada vez en forma 
# acumulada. Mostrar el resultado para diferentes interpolaciones

# In[21]:


# Por precaución ya que demora alrededor de 6 minutos en ejecutar
if False:

    t_ini = time.time()

    img = pasarAGris(imageio.imread('./imagenes/trapo.jpg'))

    alto, ancho = img.shape

    I_360_vec = img.copy()
    I_360_bil = img.copy()
    
    # Ángulo de rotación en radianes
    angulo = 10 * np.pi / 180
    
    # Matriz de rotación con respecto al origen
    T_rot_orig = np.array([[np.cos(angulo), np.sin(angulo), 0], 
                           [-np.sin(angulo), np.cos(angulo), 0], 
                           [0, 0, 1]])
    
    # Coordenadas del centro de la imagen
    i_trasl = alto/2
    j_trasl = ancho/2
    
    # Matriz de traslación hasta el origen
    T_trasl_orig = np.array([[1, 0, -i_trasl],
                            [0, 1, -j_trasl], 
                            [0, 0, 1]])
    
    # Matriz de traslación hasta el centro de la imagen
    T_trasl_centro = np.array([[1, 0 , i_trasl], 
                              [0, 1, j_trasl], 
                              [0, 0, 1]])

    # Calcule la matriz de rotación en el centro como la composición de las tres transformaciones anteriores
    T_rot_centro = T_trasl_centro @ T_rot_orig @ T_trasl_orig


    # Aplique la transformación y muestre el resultado
    for i in range(36):
        I_360_vec = transformar(I_360_vec, T_rot_centro, tipo_interpolacion='vecino', color_de_fondo=0)
        I_360_bil = transformar(I_360_bil, T_rot_centro, tipo_interpolacion='bilineal', color_de_fondo=0)

    tiempoEjecucion(t_ini)


# In[22]:


# Se levantan las imágenes ya procesadas anteriormente
I_360_vec = pasarAGris(imageio.imread('./imagenes/rotaciones/TrapoVecinoDegradada.jpg'))
I_360_bil = pasarAGris(imageio.imread('./imagenes/rotaciones/TrapoBilinealDegradada.jpg'))


# Mostrar resultados
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img)
plt.scatter(ancho/2, alto/2,c='r', marker='+')
plt.title('Imagen original')
plt.axis('off')

plt.subplot(132)
plt.imshow(I_360_vec)
plt.scatter(ancho/2, alto/2,c='r', marker='+')
plt.title('Interpolación "vecino"')
plt.axis('off')

plt.subplot(133)
plt.imshow(I_360_bil)
plt.scatter(ancho/2, alto/2,c='r', marker='+')
plt.title('Interpolación "bilineal"')
plt.axis('off')

plt.tight_layout()
plt.show()


# ##### Nota
# Como la ejecución demora bastante tiempo se optó por realizar el proceso una sola vez y guardar el resultado para levantar directamente.

# Repetir el experimento ampliando previamente la imagen por un factor por un factor de escala igual 4 y deshaciéndolo luego de concluir las rotaciones. Explicar las diferencias en el resultado.

# In[23]:


# Se levantan las imágenes ya procesadas anteriormente
trapo = pasarAGris(imageio.imread('./imagenes/trapo.jpg'))    
trapo_vecino = pasarAGris(imageio.imread('./imagenes/rotaciones/trapoReduccionVecino.jpg'))
trapo_bilineal = pasarAGris(imageio.imread('./imagenes/rotaciones/trapoReduccionBilineal.jpg'))


# Mostrar resultados
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(trapo)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(132)
plt.imshow(trapo_vecino)
plt.title('Interpolación "vecino"')
plt.axis('off')

plt.subplot(133)
plt.imshow(trapo_bilineal)
plt.title('Interpolación "bilineal"')
plt.axis('off')

plt.tight_layout()
plt.show()


# La explicación de por qué la imagen no se degrada de manera tan notoria como en la parte anterior se debe a que el proceso de interpolación se "adelanta" al agrandar la imagen, por lo que luego a la hora de rotar el error que se acarrea al realizar las interpolaciones es menor, una especie de "trampa".
# 
# ##### Nota
# Como la ejecución demora realmente mucho tiempo (orden de horas) se optó por realizar el proceso una sola vez y guardar el resultado para levantar directamente. Quizás una elección de una imagen más pequeña podría realizarse en minutos pero no lo he probado.

# ## 4) Cálculo de homografía
# Se debe implementar una función que calcule la matriz de homografia (H) para 4 **o más puntos**. 
# 
# Se sugiere utilizar funciones axuiliares de np.linalg o de otro paquete para resolver un sistema de ecuaciones o mínimos cuadrados.

# In[24]:


# Entradas:
#    puntosA   Nx2
#    puntosB   Nx2
#    con N>=4
# Salida
#    H         3x3    
def calcularHomografia(puntosA, puntosB):
    
    H, _ = cv2.findHomography(puntosA, puntosB)
    
    return H


# Otra alternativa hubiera sido aplicar una resolución por mínimos cuadrados hallando la solución que mejor aproxima la ecuación $ H.x = x'$. Existen paquetes que tienen implementados este tipo de algoritmos como $\texttt{np.linalg.lstsq}$.
# 
# Cabe destacar que sería útil la propiedad $ \quad (A.B)^t = B^t.A^t $  para dejar la incógnita del lado derecho si se utilizara np.lianlg.lstsq para que quede de la forma 
# 
# $$ x^t H^t = x^{'t}$$
# 
# - Información sobre cv2.findHomography: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# 
# - Información sobre np.linalg.lstsq: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

# ## 5)  Incrustar una imagen sobre otra
# 
# **a)** Implementar una función que dadas 2 imágenes y 4 puntos incruste una imagen sobre otra. Para esto se sugiere utilizar las ideas de las partes **3)** y **4)**.

# ### Importante
# La función que se utilizará es levemente distinta a la pedida, ya que cuando se realizó su implementación (en el práctico antes del entregable) ya se incluyó la automatización de seleccionar la cantidad de incrustaciones a realizar y la selección gráfica de los puntos para calcular las transformadas todo a la vez.

# In[25]:


# Entradas:
#    imgBase : se considera como la imagen fija
#    imgAPegar : sera la imagen que se desea pegar sobre imgBase
#    puntosPoligono 4x2 :son las 4 esquinas donde se pegara la imagen imgAPegar sobre imgBase
#    tipoInterpolacion : vecino o bilineal

# Salida
#    Devuelve la imagen imgIncrustada

# def incrustar(imgBase, imgAPegar, puntosPoligono, tipoInterpolacion='vecino'):
#     #Implementar
    
#     return imgIncrustada



### OTRA IMPLEMENTACIÓN MÁS AUTOMÁTICA

def incrustarImagen(imgBase, imgAIncrustar, nroIncrustaciones=1):
    get_ipython().run_line_magic('matplotlib', '')
    
    I_salida = imgBase.copy()
    
    for iteracion in range(nroIncrustaciones):
        plt.figure(figsize=(12,12))
        plt.imshow(I_salida, cmap='gray')
        instrucciones = 'Seleccione el polígono a incrustar.\nSeleccionar en sentido HORARIO         y empezando en el vértice superior izquierdo.'

        plt.title(instrucciones, fontsize=20, color='red')
        # tomo los puntos con ginput y dejo en formato (i,j)
        puntosDestino = np.flip(plt.ginput(n=4, timeout=60), axis=1)
        # cierro la figura creada para el ginput
        plt.show()
        plt.close()

        # dimensiones de imagenes
        M_base, N_base = imgBase.shape
        M_incr, N_incr = imgAIncrustar.shape

        puntos = np.array([[0, 0], [0, N_incr-1], [M_incr, N_incr], [M_incr-1, 0]])

        # calcular la homografía
        H = calcularHomografia(puntos, puntosDestino)
        inv_H = np.linalg.inv(H)

        # recorro la imagen base para incrustar
        for i_d in range(M_base):
                for j_d in range(N_base):
                    punto = inv_H @ [i_d, j_d, 1]
                    if abs(punto[2]) > np.finfo(float).eps:
                        punto = punto/punto[2]
                    else:
                        punto = [0, 0, 1]

                    i = punto[0]
                    j = punto[1]

                    if j >= 0 and j < N_incr-1 and i >= 0 and i < M_incr-1:
                        I_salida[i_d, j_d] = bilineal(imgAIncrustar, punto, color_de_fondo=0)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    return I_salida


# **b)** Generar una nueva imagen a partir de la imagen del Museo del Prado (Sala-pittura-olandese.jpg) donde en lugar de los cuadros, aparezca la imagen de trapo.jpg.

# In[26]:


t_ini = time.time()

imgBase = pasarAGris(imageio.imread('./imagenes/Sala-pittura-olandese.jpg'))   
imgAIncrustar = pasarAGris(imageio.imread('./imagenes/trapo.jpg'))

# Incrustar imagen
I_salida = incrustarImagen(imgBase, imgAIncrustar, nroIncrustaciones=4)

# Mostrar resultado
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(imgBase, cmap='gray')
plt.title('Imagen base')
plt.axis('off')

plt.subplot(122)
plt.imshow(imgAIncrustar, cmap='gray')
plt.title('Imagen a incrustar')
plt.axis('off')
plt.show()

plt.figure(figsize=(12,8))
plt.imshow(I_salida, cmap='gray')
plt.title('Imagen resultado')
plt.axis('off')
plt.show()

tiempoEjecucion(t_ini)


# **c)** Explicar porque en la parte anterior la frecuencia del trapo se modifica. Sugerencia: pensar qué sucede en el dominio de la frecuencia.

# **Respuesta:** 
# 
# Los artefactos que se introducen en el trapo se deben a que se esta realizando un submuestreo de la imagen para reducir su soporte, y por lo tanto también se reducirá su soporte en frecuencia. Si el soporte de destino es menor al de partida, la frecuencia máxima (Nyquist) a representar también disminuirá, por lo que en el trapo que es donde justamente se producen las oscilaciones de menor período quedan destruidas y se produce aliasing en la reconstrucción reducida de la imagen.
# 
# También es posible notar este fenómeno en las rejillas de ventilación de las paredes, para comparar y comprobar se puede visualizar la imagen original a tamaño real.

# En el siguiente ejerciccio se estudiará cómo resolver este  problema.

# ## 6) Transformada DFT
# 
# ### Submuestreo   
# **a)** Para la imagen trapo.jpg de tamaño 512x512, submuestrear %4 (tomar un pixel cada cuatro en filas y columnas y obtener imágenes 128x128). 
# 

# In[27]:


img = pasarAGris(imageio.imread('./imagenes/trapo.jpg'))

# quedarse cada 4 muestras
img_subm = img[::4, ::4]

print('Dimensiones imagen original: ', img.shape)
print('Dimensiones imagen submuestreada: ', img_subm.shape)


# **b)** Visualizar las imágenes original, submuestreada y sus transformadas DFT. Identificar las componentes de frecuencia del trapo en el espacio (determinando el período de las líneas de la tela) y su correspondiente ubicación en el dominio frecuencial. 

# In[28]:


### TRANSFORMADAS
dft_img = np.abs(fft.fftshift(fft.fft2(img)))
dft_img_subm = np.abs(fft.fftshift(fft.fft2(img_subm)))

# Mostrar imagen original y submuestreada
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')

plt.subplot(122)
plt.imshow(img_subm, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen submuestreada')

plt.tight_layout()
plt.show()


## Dibujar linea en DFT
#
# picos = [[x1 x2], 
#          [y1 y2]]
#
get_ipython().run_line_magic('matplotlib', '')
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(np.log(0.01+dft_img), cmap='gray')
plt.title('SELECCIONAR DE ARRIBA HACIA ABAJO', fontsize=20, color='red')
picos_dft = np.transpose(plt.ginput(n=2))
plt.subplot(122)
plt.imshow(np.log(0.01+dft_img_subm), cmap='gray')
picos_dftSub = np.transpose(plt.ginput(n=2))
plt.show()
plt.close()
get_ipython().run_line_magic('matplotlib', 'inline')

# calcular módulo del segmento para hallar período en la dirección
vector_dft = np.abs(picos_dft[:,0] - picos_dft[:,1])
mod_vector_dft = np.sqrt(np.sum(vector_dft**2)) / 2


vector_dftSub = np.abs(picos_dftSub[:,0] - picos_dftSub[:,1])
mod_vector_dftSub = np.sqrt(np.sum(vector_dftSub**2)) / 2



# Mostrar transformadas con segmento
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(np.log(0.01+dft_img), cmap='gray')
plt.plot(picos_dft[0], picos_dft[1], 'r')
plt.title('DFT imagen original')

plt.subplot(122)
plt.imshow(np.log(0.01+dft_img_subm), cmap='gray')
plt.plot(picos_dftSub[0], picos_dftSub[1], 'r')
plt.title('DFT imagen submuestreada')

plt.tight_layout()
plt.show()


# **c)** Explicar gráficamente por qué la frecuencia del trapo cambia al submuestrear la imagen.
# Estimar la dirección y período de las oscilaciones en la imagen submuestreada.
# 
# ### Respuesta

# Se puede apreciar en la transformada de la imagen original que se presentan dos "picos" o zonas de altos valores en ciertas componentes de frecuencia. La dirección que indica el segmento rojo es precisamente la dirección de las oscilaciones del trapo, en el límite ideal esos picos corresponderían a dos deltas, que conformarían un coseno.
# 
# Se puede observar también que para la imagen submuestreada este fenómeno ocurre pero cambiando totalmente el sentido de las oscilaciones como se puede ver en su transformada. Aquí la dirección indicada nuevamente por el segmento trazado corresponde a la nueva dirección de oscilaciones del trapo.
# 
# #### Estimar la frecuencia de las oscilaciones
# 
# Para estimar la frecuencia de las oscilaciones lo que se hará es hallar el módulo del segmento rojo y dividirlo entre 2, asumiendo que este valor coincide con el módulo del segmento que iría desde el centro de la imagen hasta uno de los dos picos de la transformada. Se aclara que los valores son aproximados ya que acarrean un error claro en la elección manual de los puntos mediante el $\textit{ginput}$.
# 
# Se calcula el período o píxeles por ciclo (p/c) mediante la posición del punto en la DFT aplicando la siguiente fórmula:
# 
# $$ T = \frac{N}{x} \hspace{2pt} p/c$$
# 
# siendo N el tamaño del soporte de la imagen en la dirección deseada, y siendo $x$ la posición de la "delta" o pico en la DFT con respecto al punto (0,0). También se puede descomponer el período tanto en el eje X o en el eje Y. Por lo tanto se expresará el resultado de la forma:
# 
# $T$ como el período en la dirección de las "deltas".
# 
# $T_x$ como el período en $p/c$ en el sentido horizontal.
# 
# $T_y$ como el período en $p/c$ en el sentido vertical.

# In[29]:


#### Imagen original

# Angulo calculado como arctan(O/A) positivo en sentido horario con EjeX
y = picos_dft[1,1] - 256
x = picos_dft[0,1] - 256
tita = np.arctan(y / x)

if x < 0 and y < 0:
    tita_grados = (tita / np.pi * 180) + 90
else:
    tita_grados = tita / np.pi * 180

largo_direccion_dft = 2*np.sqrt((np.tan(np.pi/2 - tita)*256)**2 + 256**2)

# Período de pixeles/ciclo en las distintas direcciones
T = largo_direccion_dft / mod_vector_dft
Tx = abs(512 / x)
Ty = abs(512 / y)


#### Imagen submuestreada

# Angulo calculado como arctan(O/A) positivo en sentido horario con EjeX
y_sub = picos_dftSub[1,1] - 64
x_sub = picos_dftSub[0,1] - 64
tita_sub = np.arctan(y_sub / x_sub)

if x_sub < 0 and y_sub < 0:
    tita_grados_sub = (tita_sub / np.pi * 180) + 90
else:
    tita_grados_sub = tita_sub / np.pi * 180
    
if tita_grados_sub > 90:
    largo_direccion_dftSub = 2*np.sqrt((np.tan(tita_sub)*64)**2 + 64**2)
else:
    largo_direccion_dftSub = 2*np.sqrt((np.tan(np.pi/2 - tita_sub)*64)**2 + 64**2)

# Período de pixeles/ciclo en las distintas direcciones
T_sub = largo_direccion_dftSub / mod_vector_dftSub
Tx_sub = abs(128 / x_sub)
Ty_sub = abs(128 / y_sub)


print('Imagen original \n')
print(f'Ángulo con respecto a Eje x: {tita_grados:.2f}°')
print(f'T: {T:.2f} p/c')
print(f'Tx: {Tx:.2f} p/c')
print(f'Ty: {Ty:.2f} p/c')
print('-'*25)
print('Imagen submuestreada \n')
print(f'Ángulo con respecto a Eje x: {tita_grados_sub:.2f}°')
print(f'T: {T_sub:.2f} p/c')
print(f'Tx_sub: {Tx_sub:.2f} p/c')
print(f'Ty_sub: {Ty_sub:.2f} p/c')


# **d)** Proponer e implementar una solución para los artefactos generados al submuestrear. Indicar qué sucede con las oscilaciones que resultaban problemáticas.
# 
# ### Respuesta

# Para cumplir con las hipótesis del terorema de muestreo, se debería aplicar un filtro pasabajos para eliminar aquellas frecuencias que excedan la frecuencia de Nyquist a la que se quiere submuestrear la señal (la imagen), ya que de lo contrario se produciría un efecto de Aliasing (o solapamiento) que introduciría artefactos en el resultado. Una posible solución podría ser aplicar un filtro de bluring de promedio, y luego submuestrear la imagen.

# In[30]:


def meanFilter(I, filterSize):
    
    # IMPLEMENTAR
    M, N = I.shape
    r1 = np.int(filterSize[0] / 2)
    r2 = np.int(filterSize[1] / 2)
    
    J = np.zeros((M,N))
    
    for i in np.arange(r1, M-r1):
        for j in np.arange(r2, N-r2):
            W = I[i-r1:i+r1+1, j-r2:j+r2+1]
            J[i,j] = np.mean(W)
    
    return J

def medianFilter(I, filterSize):
    
    M, N = I.shape
    r1 = np.int(filterSize[0] / 2)
    r2 = np.int(filterSize[1] / 2)
    
    J = np.zeros((M,N))
    
    for i in np.arange(r1, M-r1):
        for j in np.arange(r2, N-r2):
            W = I[i-r1:i+r1+1, j-r2:j+r2+1]
            J[i,j] = np.median(W)
    
    return J


# In[31]:


img_blur3 = medianFilter(img, (3,3))
img_blur4 = medianFilter(img, (4,4))

T = np.array([[1, 0, 0],
             [0, 1, 0],
             [0, 0, 4]])

img_reducida3 = transformar(img_blur3, T, 'bilineal', color_de_fondo=0)
img_reducida4 = transformar(img_blur4, T, 'bilineal', color_de_fondo=0)

img_reducida3 = img_reducida3[:128,:128]
img_reducida4 = img_reducida4[:128,:128]

plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(img_subm, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen submuestreada %4 sin pasabajos')
plt.axis('off')

plt.subplot(122)
plt.imshow(img_reducida3, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen reducida con filtro de media (3,3)')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(img_subm, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen submuestreada %4 sin pasabajos')
plt.axis('off')

plt.subplot(122)
plt.imshow(img_reducida4, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen reducida con filtro de media (4,4)')
plt.axis('off')

plt.tight_layout()
plt.show()


# Como se aprecia en las imágenes, haberle aplicado el filtro pasabajos previo al submuestreo hace que la imagen resultado sea más suave y no contenga tantos artefactos como en la imagen de la izquierda.
# 
# Sin embargo, en la región del trapo se observa que persiste el problema con la frecuencia de oscilaciones o el patrón que contenía en la imagen original, esto se debe simplemente a que dichas oscilaciones correspondían a un soporte superior (512x512) y por lo tanto era posible representarlas. En el nuevo soporte (128x128) el período de oscilación máximo que se puede representar es inferior al original, por lo que al haberle aplicado el filtro pasabajos estas oscilaciones mueren y ya no es posible conservarlas en la imagen de destino.

# ## 7) Template matching
# 
# **a)** Diseñar un programa que detecte automáticamente cuántas veces aparece Wally en la imagen “dondeEstaWally.png”.  Se deberá implementar la comparación de al menos tres maneras diferentes:
# 
# * correlación espacial
# * correlación espacial implementado mediante FFT
# * correlación de fase ([phase correlation](https://en.wikipedia.org/wiki/Phase_correlation))

# In[32]:


def filtroPrewitt(I):
    
    M, N = I.shape
    Gx = np.zeros((M, N))
    Gy = np.zeros((M, N))
    
    gradGx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    gradGy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    
    Gx = np.array([[np.sum(gradGx * I[i-1:i+2, j-1:j+2]) for j in range(1,N-1)] for i in range(1, M-1)])
    Gy = np.array([[np.sum(gradGy * I[i-1:i+2, j-1:j+2]) for j in range(1,N-1)] for i in range(1, M-1)])

    G = np.sqrt(Gx**2 + Gy**2).astype('uint8')
    
    return G


# In[33]:


wally = pasarAGris(imageio.imread('./imagenes/Wally_template.png'))   
dondeEstaWally = pasarAGris(imageio.imread('./imagenes/dondeEstaWally.png'))

# Mostrar resultados
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.imshow(wally, cmap='gray', vmin=0, vmax=255)
plt.title('Wally')
plt.axis('off')

plt.subplot(122)
plt.imshow(dondeEstaWally, cmap='gray', vmin=0, vmax=255)
plt.title('¿Dónde está Wally?')
plt.axis('off')

plt.tight_layout()
plt.show()


# **b)** Comparar los resultados y los tiempos insumidos por los distintos métodos

# ## Espacio

# In[34]:


t_ini = time.time()

wallyT = np.flip(wally).astype(float) - np.mean(wally)
dondeEstaWally_ = dondeEstaWally.astype(float) - np.mean(dondeEstaWally)

aquiEsta = signal.convolve(dondeEstaWally_, wallyT, mode='same', method='fft')
aquiEsta = aquiEsta / np.max(aquiEsta)* 255
aquiEstaESP = aquiEsta.copy()
aquiEsta[aquiEsta < 140] = 0

# Puntos para marcar donde esta Wally
r, c = np.where(aquiEsta > 0)

# Marcar posiciones de Wally
plt.figure(figsize=(12, 12))
plt.imshow(dondeEstaWally, cmap='gray', vmin=0, vmax=255)
for i in range(r.size):
    plt.scatter(c[i], r[i], color='r')
plt.title('Puntos donde se "detecta" a Wally')
plt.axis('off')
plt.show()

tiempoEjecucion(t_ini)


# ## Frecuencia

# In[35]:


wallyExt = np.zeros_like(dondeEstaWally)
wally_ = wally.astype(float) - np.mean(wally)

## Generar imagen de wally para transformar
M, N = wally.shape

M_ini = int(M/2)
N_ini = int(N/2)

# Si el ancho o alto es impar
if M%2:
    M_fin = int(M/2) + 1
else:
    M_fin = int(M/2)
    
if N%2:
    N_fin = int(N/2) + 1
else:
    N_fin = int(N/2)
    
# Asignar los cuadrantes de wally en las esquinas
wallyExt[:M_fin, :N_fin] = wally[-M_fin:, -N_fin:]
wallyExt[-M_ini:, :N_fin] = wally[:M_ini, -N_fin:]
wallyExt[:M_fin, -N_ini:] = wally[-M_fin:, :N_ini]
wallyExt[-M_ini:, -N_ini:] = wally[:M_ini, :N_ini]

# Restar el promedio
wallyExt_ = wallyExt.astype(float) - np.mean(wallyExt)
wallyExtFlip_ = np.flip(wallyExt_)

## mostrar imagen
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(wallyExt_, vmin=0, vmax=255)
plt.title('Wally extendido')
plt.axis('off')
plt.subplot(122)
plt.imshow(wallyExtFlip_, vmin=0, vmax=255)
plt.title('Wally extendido y simetrizado para transformar')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[36]:


t_ini = time.time()

## transformadas
wallyDFT = fft.fft2(wallyExtFlip_)
dondeEstaWallyDFT = fft.fft2(dondeEstaWally_)

# Multiplicación en frecuencia
aquiEstaDFT = dondeEstaWallyDFT * wallyDFT

aquiEsta = np.real(fft.ifft2(aquiEstaDFT))
aquiEsta = (aquiEsta/np.max(aquiEsta)) *255

# Detectar los bordes como herramienta para derivar y resaltar los puntos 
aquiEsta = filtroPrewitt(aquiEsta)
# Normalizar en [0, 255]
aquiEsta = 255* ( (aquiEsta - np.min(aquiEsta)) / (np.max(aquiEsta) - np.min(aquiEsta)))
# Umbralizar
aquiEsta[aquiEsta<190] = 0

# Puntos para marcar donde esta Wally
r, c = np.where(aquiEsta > 0)

# Marcar posiciones de Wally
plt.figure(figsize=(12, 12))
plt.imshow(dondeEstaWally, cmap='gray', vmin=0, vmax=255)
for i in range(r.size):
    plt.scatter(c[i], r[i], color='r')
plt.title('Puntos donde se "detecta" a Wally')
plt.axis('off')
plt.show()

tiempoEjecucion(t_ini)


# Como la salida de la antitransformada del producto no tenía tan claros los picos donde se producía la correlación más alta como sí ocurre trabajando en el espacio dual, se optó por un post-procesamiento que incluyó resaltar los picos a partir de la derivada, y para ello se utilizó el detector de bordes Prewitt implementado en el práctico 2. También se realizó una etapa de normalización para llevar los valores al rango [0, 255]. Es por eso que el tiempo consumido para ejecutar esta parte es considerablemente superior al de la anterior.

# ## Fase

# In[37]:


# Calculo de ventana Hamming
M, N = wallyExt.shape

hamm_M = scipy.hamming(M)
hamm_M = np.reshape(hamm_M, (M,1))

hamm_N = scipy.hamming(N)
hamm_N = np.reshape(hamm_N, (N,1))

window = hamm_M @ hamm_N.T

# multiplicar las imágenes por la ventana
wallyExtWin = window * wallyExt
dondeEstaWallyWin = window * dondeEstaWally


## transformadas
wallyDFT = fft.fft2(wallyExtWin)
dondeEstaWallyDFT = fft.fft2(dondeEstaWallyWin)

# Calcular el producto y normalización
aquiEstaDFT = np.conjugate(wallyDFT) * dondeEstaWallyDFT
aquiEstaDFT /= np.abs(aquiEstaDFT)

# reconstruir y umbralizar
aquiEsta = np.real(fft.ifft2(aquiEstaDFT))
aquiEsta = (aquiEsta/np.max(aquiEsta))*255
aquiEsta[aquiEsta < 20] = 0
aquiEsta[aquiEsta > 0] = 255

# Puntos para marcar donde esta Wally
r, c = np.where(aquiEsta > 0)

# Marcar posiciones de Wally
plt.figure(figsize=(12, 12))
plt.imshow(dondeEstaWally, cmap='gray', vmin=0, vmax=255)
for i in range(r.size):
    plt.scatter(c[i], r[i], color='r')
plt.title('Puntos donde se "detecta" a Wally')
plt.axis('off')
plt.show()


# Este método fue el más preciso y claro de los tres implementados, con la salvedad que para una de las ubicaciones de Wally la intensidad de la correlación era apenas perceptible, pero de todas maneras fue posible establecer un umbral para separar las ubicaciones.
# 
# En las tres instancias se utilizaron marcadores en las posiciones distintas a cero luego de umbralizar cada detección. Como el segundo caso (Frecuencia) fue el más problemático para detectar los picos, se observa que aparece más de un marcador por cada posición de Wally.
