#!/usr/bin/env python
# coding: utf-8

# # Tratamiento de imágenes - 2020 - Entregable 2
# 
# # Fecha de entrega: 28/10/2020

# **Importante:** En todos los ejercicios se espera que se hagan comentarios sobre decisiones tomadas en la implementación así como un análisis de los resultados. Estos comentarios y análisis se pueden entregar en secciones agregadas a los notebooks o en un informe aparte. En caso de entregar un notebook, estos deben estar guardados con el resultado de su ejecución.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio

import scipy as sp
from scipy import fftpack as fft

import cv2
import skimage

from skimage.restoration import denoise_nl_means, estimate_sigma 

import time

# Parametros plot
plt.rcParams["image.cmap"] = 'gray'


# In[ ]:


## Función auxiliar para imprimir el tiempo de ejecución
def tiempoEjecucion(t_ini):
    print(f'Tiempo de ejecución aproximado: {(time.time()-t_ini):.2f}s')
    
## Función auxiliar para convertir imágenes a escala de grises
def pasarAGris(img):
    img_ = img.copy()
    
    if img_.ndim == 3:
        img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    
    return img_


# ## Restauración

# ###  1) Restauración de imágenes con ruido

# **a) A partir de una imagen original en escala de grises, generar imágenes con diferentes niveles de ruido gaussiano (por ejemplo σ= 10, 20, 40) y ruido “sal y pimienta” (por ejemplo p=0.1, 0.4, 0.8).**
#   
# Otras [imágenes estándar](http://en.wikipedia.org/wiki/Standard_test_image) se pueden encontrar en http://sipi.usc.edu/database/database.php?volume=misc    

# In[ ]:


def addGaussianNoise(I,sigma):
    
    M, N = I.shape
    Noise = np.random.normal(0, sigma, size=(M,N))
    J = I.copy().astype('float64')
    
    J += Noise
    
    return J.astype(int)

def addSaltPepperNoise(I,p):
 
    M, N = I.shape
    J = I.copy()

    # add salt
    coords_s = np.array([np.random.randint(0, N, int(N*p/2)) for i in range(M)])
    coords_p = np.array([np.random.randint(0, N, int(N*p/2)) for i in range(M)])
    
    for i in range(M):
        J[i,coords_s[i,:]] = 255
        J[i,coords_p[i,:]] = 0
    
    return J


# In[ ]:


Files_misc = ['./imagenes/5.1.11.tiff', './imagenes/5.1.12.tiff']

img_avion = imageio.imread(Files_misc[0])
img_reloj = imageio.imread(Files_misc[1])

avion_Gauss10 = addGaussianNoise(img_avion, 10)
avion_Gauss20 = addGaussianNoise(img_avion, 20)
avion_Gauss40 = addGaussianNoise(img_avion, 40)

reloj_Salt1 = addSaltPepperNoise(img_reloj, 0.1)
reloj_Salt4 = addSaltPepperNoise(img_reloj, 0.4)
reloj_Salt8 = addSaltPepperNoise(img_reloj, 0.8)

plt.figure(figsize=(20,12))
plt.subplot(241)
plt.imshow(img_avion, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(242)
plt.imshow(avion_Gauss10, cmap='gray', vmin=0, vmax=255)
plt.title('Ruido gaussiano $\sigma = 10$')
plt.axis('off')

plt.subplot(243)
plt.imshow(avion_Gauss20, cmap='gray', vmin=0, vmax=255)
plt.title('Ruido gaussiano $\sigma = 20$')
plt.axis('off')

plt.subplot(244)
plt.imshow(avion_Gauss40, cmap='gray', vmin=0, vmax=255)
plt.title('Ruido gaussiano $\sigma = 40$')
plt.axis('off')

plt.subplot(245)
plt.imshow(img_reloj, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(246)
plt.imshow(reloj_Salt1, cmap='gray', vmin=0, vmax=255)
plt.title('Ruido sal y pimienta $p = 0.1$')
plt.axis('off')

plt.subplot(247)
plt.imshow(reloj_Salt4, cmap='gray', vmin=0, vmax=255)
plt.title('Ruido sal y pimienta $p = 0.4$')
plt.axis('off')

plt.subplot(248)
plt.imshow(reloj_Salt8, cmap='gray', vmin=0, vmax=255)
plt.title('Ruido sal y pimienta $p = 0.8$')
plt.axis('off')


plt.tight_layout()
plt.show()


# **b) Procesar las imágenes de (a) con**
# * filtros de media 
# * filtro de media adaptivo (implementar el filtro y un criterio para estimar la varianza del ruido)
# * filtros de mediana
# * Non local means (usar implementación de OpenCV y/o de scikit-image)

# In[ ]:


def meanFilter(I, filterSize):
    
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


# In[ ]:


def meanAdaptiveFilter(I, filterSize, var_n):
    
    M, N = I.shape
    r1 = np.int(filterSize[0] / 2)
    r2 = np.int(filterSize[1] / 2)
    
    J = np.zeros((M,N))
    
    for i in np.arange(r1, M-r1):
        for j in np.arange(r2, N-r2):
            W = I[i-r1:i+r1+1, j-r2:j+r2+1]
            u = np.mean(W)
            var_i = np.mean((W-u)**2)
            
            J[i,j] = u + abs((var_i - var_n)/var_i)*(I[i,j] - u)
            
    return J


# #### Filtro de media y mediana

# In[ ]:


filterSize = 5
filt = np.array((filterSize, filterSize))

# Imágenes con filtro de media y mediana
avion10_m5 = meanFilter(avion_Gauss10, filt)
avion10_M5 = medianFilter(avion_Gauss10, filt)
avion20_m5 = meanFilter(avion_Gauss20, filt)
avion20_M5 = medianFilter(avion_Gauss20, filt)
avion40_m5 = meanFilter(avion_Gauss40, filt)
avion40_M5 = medianFilter(avion_Gauss40, filt)
reloj1_m5 = meanFilter(reloj_Salt1, filt)
reloj1_M5 = medianFilter(reloj_Salt1, filt)
reloj4_m5 = meanFilter(reloj_Salt4, filt)
reloj4_M5 = medianFilter(reloj_Salt4, filt)

imagenesFiltradas = [avion_Gauss10, avion_Gauss40, reloj_Salt1, reloj_Salt4, avion10_m5, avion40_m5, reloj1_m5,
                    reloj4_m5, avion10_M5, avion40_M5, reloj1_M5, reloj4_M5]
titulosPlot = ['Ruido gaussiano $\sigma = 10$', 'Ruido gaussiano $\sigma = 40$', 'Ruido S&P $p = 0.1$',
              'Ruido S&P $p = 0.4$', f'Filtro media {filt}', f'Filtro media {filt}', f'Filtro media {filt}',
              f'Filtro media {filt}', f'Filtro mediana {filt}', f'Filtro mediana {filt}',
              f'Filtro mediana {filt}', f'Filtro mediana {filt}']


plt.figure(figsize=(20,20))
for i in range(12):
    plt.subplot(4,4,i+1)
    plt.imshow(imagenesFiltradas[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titulosPlot[i])
    plt.axis('off')   
plt.tight_layout()
plt.show()


# #### Filtro de media adaptivo

# In[ ]:


def plotSquare():
    left, width = .5, .1
    bottom, height = .5, .1
    ax = plt.gca()
    p = plt.Rectangle((left, bottom), width, height, fill=False,color='r')
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)


# In[ ]:


filterSize = 5
filt = (filterSize, filterSize)

# Ventanas usadas para calcular la varianza del ruido
W_avion10 = avion_Gauss10[102:128, 128:154]
W_avion20 = avion_Gauss20[102:128, 128:154]
W_avion40 = avion_Gauss40[102:128, 128:154]

# Cálculo de la varianza del ruido
sigma10 = np.mean((W_avion10 - np.mean(W_avion10))**2)
sigma20 = np.mean((W_avion20 - np.mean(W_avion20))**2)
sigma40 = np.mean((W_avion40 - np.mean(W_avion40))**2)

# Imagenes con filtro de media adaptivo
avion10_m5Ad = meanAdaptiveFilter(avion_Gauss10, filt, sigma10)
avion20_m5Ad = meanAdaptiveFilter(avion_Gauss20, filt, sigma20)
avion40_m5Ad = meanAdaptiveFilter(avion_Gauss40, filt, sigma40)


figura = plt.figure(figsize=(20,20))
plt.subplot(331)
plt.imshow(avion_Gauss10, cmap='gray', vmin=0, vmax=255)
plotSquare()
plt.title('Imagen ruidosa y ventana utilizada para calcular $\sigma_n^2$')
plt.axis('off')

plt.subplot(332)
plt.imshow(avion_Gauss20, cmap='gray', vmin=0, vmax=255)
plotSquare()
plt.title('Imagen ruidosa y ventana utilizada para calcular $\sigma_n^2$')
plt.axis('off')

plt.subplot(333)
plt.imshow(avion_Gauss40, cmap='gray', vmin=0, vmax=255)
plotSquare()
plt.title('Imagen ruidosa y ventana utilizada para calcular $\sigma_n^2$')
plt.axis('off')

plt.subplot(334)
plt.imshow(avion10_m5, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen con filtro de media de tamaño {filt}')
plt.axis('off')

plt.subplot(335)
plt.imshow(avion20_m5, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen con filtro de media de tamaño {filt}')
plt.axis('off')

plt.subplot(336)
plt.imshow(avion40_m5, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen con filtro de media de tamaño {filt}')
plt.axis('off')

plt.subplot(337)
plt.imshow(avion10_m5Ad, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen con filtro de media adaptivo de tamaño {filt}')
plt.axis('off')

plt.subplot(338)
plt.imshow(avion20_m5Ad, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen con filtro de media adaptivo de tamaño {filt}')
plt.axis('off')

plt.subplot(339)
plt.imshow(avion40_m5Ad, cmap='gray', vmin=0, vmax=255)
plt.title(f'Imagen con filtro de media adaptivo de tamaño {filt}')
plt.axis('off')

plt.tight_layout()
plt.show()


# #### Non local means

# In[ ]:


# Utilizando skimage

# sigma estimado de las imagenes ruidosas
sigmaAvion10 = estimate_sigma(avion_Gauss10)
sigmaAvion20 = estimate_sigma(avion_Gauss20)
sigmaAvion40 = estimate_sigma(avion_Gauss40)

avion10_NlMeans = denoise_nl_means(avion_Gauss10.astype(float), h=0.3*sigmaAvion10, 
                                   sigma=np.sqrt(sigma10), fast_mode=True)
avion20_NlMeans = denoise_nl_means(avion_Gauss20.astype(float), h=0.3*sigmaAvion20, 
                                   sigma=np.sqrt(sigma20), fast_mode=True)
avion40_NlMeans = denoise_nl_means(avion_Gauss40.astype(float), h=0.3*sigmaAvion40, 
                                   sigma=np.sqrt(sigma40), fast_mode=True)
reloj4_NlMeans = denoise_nl_means(reloj_Salt4.astype(float), h=10, fast_mode=True)


plt.figure(figsize=(18,11))

plt.subplot(241)
plt.imshow(avion_Gauss10, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen + Ruido gaussiano $\sigma = 10$')
plt.axis('off')

plt.subplot(242)
plt.imshow(avion_Gauss20, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen + Ruido gaussiano $\sigma = 20$')
plt.axis('off')

plt.subplot(243)
plt.imshow(avion_Gauss40, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen + Ruido gaussiano $\sigma = 40$')
plt.axis('off')

plt.subplot(244)
plt.imshow(reloj_Salt4, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen + Ruido S&P $p = 0.4$')
plt.axis('off')

plt.subplot(245)
plt.imshow(avion10_NlMeans, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen restaurada NlMeans')
plt.axis('off')

plt.subplot(246)
plt.imshow(avion20_NlMeans, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen restaurada NlMeans')
plt.axis('off')

plt.subplot(247)
plt.imshow(avion40_NlMeans, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen restaurada NlMeans')
plt.axis('off')

plt.subplot(248)
plt.imshow(reloj4_NlMeans, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen restaurada NlMeans')
plt.axis('off')

plt.suptitle('Skimage', fontsize=20)

plt.tight_layout()
plt.show()


# **d) Cuantificar la reducción de ruido utilizando medidas adecuadas (PSNR, RMSE, etc,).**

# In[ ]:


def PSNR(I,Iruidosa):
    
    MSE = np.sum((I - Iruidosa)**2) / (I.size-1)
    
    MAX_I = 255
    
    PSNR = 10*np.log10(MAX_I**2 / MSE)
    
    return PSNR

def RMSE(I,Iruidosa):
    
    MSE = np.sum((I - Iruidosa)**2) / (I.size-1)
    
    return np.sqrt(MSE)


# In[ ]:


# Cálculo de medidas para imagen de avión
PSNR_Avion_Ruido10 = PSNR(img_avion, avion_Gauss10)
PSNR_Avion_Ruido20 = PSNR(img_avion, avion_Gauss20)
PSNR_Avion_Ruido40 = PSNR(img_avion, avion_Gauss40)
PSNR_Avion10_m5 = PSNR(img_avion, avion10_m5)
PSNR_Avion20_m5 = PSNR(img_avion, avion20_m5)
PSNR_Avion40_m5 = PSNR(img_avion, avion40_m5)
PSNR_Avion10_M5 = PSNR(img_avion, avion10_M5)
PSNR_Avion20_M5 = PSNR(img_avion, avion20_M5)
PSNR_Avion40_M5 = PSNR(img_avion, avion40_M5)
PSNR_Avion10_m5Ad = PSNR(img_avion, avion10_m5Ad)
PSNR_Avion20_m5Ad = PSNR(img_avion, avion20_m5Ad)
PSNR_Avion40_m5Ad = PSNR(img_avion, avion40_m5Ad)
PSNR_Avion10_NLM = PSNR(img_avion, avion10_NlMeans)
PSNR_Avion20_NLM = PSNR(img_avion, avion20_NlMeans)
PSNR_Avion40_NLM = PSNR(img_avion, avion40_NlMeans)

RMSE_Avion_Ruido10 = RMSE(img_avion, avion_Gauss10)
RMSE_Avion_Ruido20 = RMSE(img_avion, avion_Gauss20)
RMSE_Avion_Ruido40 = RMSE(img_avion, avion_Gauss40)
RMSE_Avion10_m5 = RMSE(img_avion, avion10_m5)
RMSE_Avion20_m5 = RMSE(img_avion, avion20_m5)
RMSE_Avion40_m5 = RMSE(img_avion, avion40_m5)
RMSE_Avion10_M5 = RMSE(img_avion, avion10_M5)
RMSE_Avion20_M5 = RMSE(img_avion, avion20_M5)
RMSE_Avion40_M5 = RMSE(img_avion, avion40_M5)
RMSE_Avion10_m5Ad = RMSE(img_avion, avion10_m5Ad)
RMSE_Avion20_m5Ad = RMSE(img_avion, avion20_m5Ad)
RMSE_Avion40_m5Ad = RMSE(img_avion, avion40_m5Ad)
RMSE_Avion10_NLM = RMSE(img_avion, avion10_NlMeans)
RMSE_Avion20_NLM = RMSE(img_avion, avion20_NlMeans)
RMSE_Avion40_NLM = RMSE(img_avion, avion40_NlMeans)

# Cálculo de medidas para imagen de reloj
PSNR_Reloj1_Ruido = PSNR(img_reloj, reloj_Salt1)
PSNR_Reloj4_Ruido = PSNR(img_reloj, reloj_Salt4)
PSNR_Reloj8_Ruido = PSNR(img_reloj, reloj_Salt8)
PSNR_Reloj1_m5 = PSNR(img_reloj, reloj1_m5)
PSNR_Reloj4_m5 = PSNR(img_reloj, reloj4_m5)
PSNR_Reloj1_M5 = PSNR(img_reloj, reloj1_M5)
PSNR_Reloj4_M5 = PSNR(img_reloj, reloj4_M5)
PSNR_Reloj4_NLM = PSNR(img_reloj, reloj4_NlMeans)

RMSE_Reloj1_Ruido = RMSE(img_reloj, reloj_Salt1)
RMSE_Reloj4_Ruido = RMSE(img_reloj, reloj_Salt4)
RMSE_Reloj8_Ruido = RMSE(img_reloj, reloj_Salt8)
RMSE_Reloj1_m5 = RMSE(img_reloj, reloj1_m5)
RMSE_Reloj4_m5 = RMSE(img_reloj, reloj4_m5)
RMSE_Reloj1_M5 = RMSE(img_reloj, reloj1_M5)
RMSE_Reloj4_M5 = RMSE(img_reloj, reloj4_M5)
RMSE_Reloj4_NLM = RMSE(img_reloj, reloj4_NlMeans)


# In[ ]:


print('Avión con ruido Gaussiano σ = 10 \t \t    |   Reloj con ruido S&P p = 0.1')
print('-'*52 + '|' + '-'*54)
print(f'Ruidosa \tPSNR = {PSNR_Avion_Ruido10:.2f} dB \tRMSE = {RMSE_Avion_Ruido10:.2f}|\tRuidosa \tPSNR = {PSNR_Reloj1_Ruido:.2f} dB ' + f'\tRMSE = {RMSE_Reloj1_Ruido:.2f}')
print(f'Filtro m5 \tPSNR = {PSNR_Avion10_m5:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_m5:.2f}|\tFiltro m5 \tPSNR = {PSNR_Reloj1_m5:.2f} dB ' + f'\tRMSE = {RMSE_Reloj1_m5:.2f}')
print(f'Filtro m5Ad \tPSNR = {PSNR_Avion10_m5Ad:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_m5Ad:.2f}|')
print(f'Filtro M5 \tPSNR = {PSNR_Avion10_M5:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_M5:.2f}|\tFiltro M5 \tPSNR = {PSNR_Reloj1_M5:.2f} dB ' + f'\tRMSE = {RMSE_Reloj1_M5:.2f}')
print(f'Filtro NLM \tPSNR = {PSNR_Avion10_NLM:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_NLM:.2f} |')
print()
print('Avión con ruido Gaussiano σ = 20 \t \t    |   Reloj con ruido S&P p = 0.4')
print('-'*52 + '|' + '-'*54)
print(f'Ruidosa \tPSNR = {PSNR_Avion_Ruido10:.2f} dB \tRMSE = {RMSE_Avion_Ruido10:.2f}|\tRuidosa \tPSNR = {PSNR_Reloj4_Ruido:.2f} dB ' + f'\tRMSE = {RMSE_Reloj4_Ruido:.2f}')
print(f'Filtro m5 \tPSNR = {PSNR_Avion10_m5:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_m5:.2f}|\tFiltro m5 \tPSNR = {PSNR_Reloj4_m5:.2f} dB ' + f'\tRMSE = {RMSE_Reloj4_m5:.2f}')
print(f'Filtro m5Ad \tPSNR = {PSNR_Avion10_m5Ad:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_m5Ad:.2f}|')
print(f'Filtro M5 \tPSNR = {PSNR_Avion10_M5:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_M5:.2f}|\tFiltro M5 \tPSNR = {PSNR_Reloj4_M5:.2f} dB ' + f'\tRMSE = {RMSE_Reloj4_M5:.2f}')
print(f'Filtro NLM \tPSNR = {PSNR_Avion10_NLM:.2f} dB ' + f'\tRMSE = {RMSE_Avion10_NLM:.2f} |')
print()
print('Avión con ruido Gaussiano σ = 40 \t \t    |')
print('-'*52 + '|')
print(f'Ruidosa \tPSNR = {PSNR_Avion_Ruido40:.2f} dB ' + f'\tRMSE = {RMSE_Avion_Ruido40:.2f}|')
print(f'Filtro m5 \tPSNR = {PSNR_Avion40_m5:.2f} dB ' + f'\tRMSE = {RMSE_Avion40_m5:.2f}|')
print(f'Filtro m5Ad \tPSNR = {PSNR_Avion40_m5Ad:.2f} dB ' + f'\tRMSE = {RMSE_Avion40_m5Ad:.2f}|')
print(f'Filtro M5 \tPSNR = {PSNR_Avion40_M5:.2f} dB ' + f'\tRMSE = {RMSE_Avion40_M5:.2f}|')
print(f'Filtro NLM \tPSNR = {PSNR_Avion40_NLM:.2f} dB ' + f'\tRMSE = {RMSE_Avion40_NLM:.2f} |')


# Las medidas para los filtros de media dan bastante extrañas, uno supondría que al filtrar y reducir el ruido la relacion de PSNR debería aumentar algunos dB con respecto a la imagen ruidosa, pero esto no ocurre.
# 
# Sin embargo, para el algoritmo de NLM sí se observa un despegue en decibeles de las medididas de PSNR y una reducción bastante drástica del error.

# ### 2) Movimiento lineal uniforme 

# **a) Ver PSF y MTF del movimiento lineal uniforme (se provee una función que calcula la PSF a partir del largo y el ángulo del movimiento)**

# In[ ]:


from skimage import transform
import numpy as np

def motionPSF(largo=9,angulo_en_grados=0, sz=65):
    # Se genera una imagen cuadrada de lado sz y se pinta una linea horizontal 
    # en el medio de la imagen de tamaño igual al largo
    f = np.zeros((sz,sz)) 
    f[sz // 2, sz // 2 - largo//2 : sz // 2 + largo//2 + 1]=1  # "//" hace una división entera  
    # Luego se rota el angulo especificado
    f = transform.rotate(f,angulo_en_grados);
    # Finalmente se normaliza
    f = f/np.sum(f);
    return f


# In[ ]:


def matchSize(img1_, img2):
    '''Se asume que img1 = (mxn), img2 = (MxN) y que M > m, N > n'''
    
    img1 = img1_.copy()
    M, N = img2.shape
    m, n = img1.shape
    
    difancho = (N-n)
    difalto = M-m
    anchoL= (N - n) // 2 
    altoD = (M - m) // 2
    
    if not difancho%2:
        anchoR = anchoL
    else:
        anchoR = anchoL+1
    if not difalto%2:
        altoU = altoD
    else:
        altoU = altoD+1
       
    cerosL = np.zeros((M, anchoL))
    cerosR = np.zeros((M, anchoR))
    cerosD = np.zeros((altoD, n))
    cerosU = np.zeros((altoU, n))
    
    img1 = np.vstack((cerosU, img1, cerosD))
    img1 = np.hstack((cerosL, img1, cerosR))
    
    return img1


# **b) Se busca identificar el número de matrícula  de la imagen blurred_noisy_car.png. Intente reconocer el número utilizando las siguientes técnicas:**
# 
# * Filtrado inverso (implementar)   
# * Filtrado pseudo inverso (implementar)    
# * Filtrado de Wiener (usar la función “wiener” del paquete skimage.restoration con diversos parámetros)

# In[ ]:


def deblur(img, PSF, cte=3e-2, plot=False):
    
    def shift(I):
        return fft.fftshift(I)
    
    
    IMG = fft.fft2(fft.fftshift(img))
    H = fft.fft2(matchSize(PSF,img))
    G = (1/H)*(1/(1+(cte/np.abs(H)**2)))
    
    IMG_Hat = IMG * G
    img_Hat = np.real(fft.ifft2(IMG_Hat))
    
    if plot:
        plt.figure(figsize=(12,8))
        plt.imshow(np.log(0.01+abs(shift(IMG))))
        plt.title('IMG(f)')
        plt.show()

        plt.figure(figsize=(12,8))
        plt.imshow(np.log(0.01+abs(shift(H))))
        plt.title('H(f)')
        plt.show()

        plt.figure(figsize=(12,8))
        plt.imshow(np.log(0.01+abs(shift(G))))
        plt.title('G(f)')
        plt.show()

        plt.figure(figsize=(12,8))
        plt.imshow(np.log(0.01+abs(shift(IMG_Hat))))
        plt.title('$\hat{IMG}$(f)')
        plt.show()
    
    return img_Hat


# In[ ]:


matricula = imageio.imread('./imagenes/blurred_noisy_car.png')
if matricula.ndim == 3:
    matricula = cv2.cvtColor(matricula, cv2.COLOR_RGB2GRAY)

PSF = motionPSF(largo=20, angulo_en_grados=15)
PSFmatch = matchSize(PSF, matricula)
matriculaHatInverso = deblur(matricula,PSF,cte=0)
matriculaHatPseudo = deblur(matricula, PSF, cte=1e-4)

def Wiener(bal=1e-4):
    return skimage.restoration.wiener(matricula, PSFmatch, balance=bal, clip=False)

matriculaWiener = Wiener(1e-4)

# IMAGEN ORIGINAL Y FILTRO INVERSO
plt.figure(figsize=(16,6))
plt.subplot(121)
plt.imshow(matricula, vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')
plt.subplot(122)
plt.imshow(matriculaHatInverso, vmin=0, vmax=255)
plt.title('Imagen con filtro inverso')
plt.axis('off')
plt.tight_layout()
plt.show()

# FILTRO PSEUDO INVERSO
plt.figure(figsize=(16,12))
plt.subplot(221)
plt.imshow(deblur(matricula,PSF,cte=0.1), vmin=0, vmax=255)
plt.title('cte = 0.1')
plt.axis('off')
plt.subplot(222)
plt.imshow(deblur(matricula,PSF,cte=1e-2), vmin=0, vmax=255)
plt.title('cte = 0.01')
plt.axis('off')
plt.subplot(223)
plt.imshow(deblur(matricula,PSF,cte=1e-3), vmin=0, vmax=255)
plt.title('cte = 0.001')
plt.axis('off')
plt.subplot(224)
plt.imshow(matriculaHatPseudo, vmin=0, vmax=255)
plt.title('cte = 0.0001')

plt.suptitle('Imagen con filtro pseudo inverso', fontsize=18, color='r')
plt.axis('off')
plt.tight_layout()
plt.show()

## FILTRO DE WIENER
plt.figure(figsize=(16,12))
plt.subplot(221)
plt.imshow(Wiener(0.1), vmin=0, vmax=255)
plt.title('cte = 0.1')
plt.axis('off')
plt.subplot(222)
plt.imshow(Wiener(1e-2), vmin=0, vmax=255)
plt.title('cte = 0.01')
plt.axis('off')
plt.subplot(223)
plt.imshow(Wiener(1e-3), vmin=0, vmax=255)
plt.title('cte = 0.001')
plt.axis('off')
plt.subplot(224)
plt.imshow(matriculaWiener, vmin=0, vmax=255)
plt.title('cte = 0.0001')

plt.suptitle('Imagen con filtro Wiener del paquete Skimage', fontsize=18, color='r')
plt.axis('off')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,6))
plt.subplot(121)
plt.imshow(matriculaHatPseudo, vmin=0, vmax=255)
plt.title('Filtrado pseudo inverso')
plt.axis('off')
plt.subplot(122)
plt.imshow(matriculaWiener, vmin=0, vmax=255)
plt.title('Filtrado de Wiener')
plt.axis('off')
plt.suptitle('Comparación entre ambos resultados', fontsize=18)
plt.tight_layout()
plt.show()


# ## Segmentación
# ### 3) Detección de líneas
# 

# En esta parte se pedirá que trabaje con una foto de su cédula de identidad completamente contenida dentro de la foto, tomada sobre un fondo relativamente oscuro, y que aparezca con una cantidad de rotación y perspectiva moderada (hasta 30 grados de rotación y de modo que el largo de lados paralelos de la cédula no difieran más de un 20%). Se sugiere que la foto sea de aproximadamente 1 megapixeles.

# a) Agregar ruido gaussiano a la imagen tomada de la cédula de identidad (con σ= 30)

# In[ ]:


cedula = pasarAGris(imageio.imread('./imagenes/cedula3.jpg'))
cedula2 = pasarAGris(imageio.imread('./imagenes/cedulaTimag2.jpg'))

cedulaNoise = addGaussianNoise(cedula, 30)
cedula2Noise = addGaussianNoise(cedula2, 30)

imgPrueba = pasarAGris(imageio.imread('./imagenes/pruebaHough.jpg'))
imgPrueba2 = pasarAGris(imageio.imread('./imagenes/pruebaHough2.jpg'))

plt.figure(figsize=(12,7))
plt.subplot(121)
plt.imshow(cedula)
plt.title('Imagen 1')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedula2)
plt.title('Imagen 2')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,7))
plt.subplot(121)
plt.imshow(cedulaNoise)
plt.title('Imagen 1 + ruido')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedula2Noise)
plt.title('Imagen 2 + ruido')
plt.axis('off')
plt.tight_layout()
plt.show()


# b) Implementar detección de líneas con la transformada de Hough de manera que los bordes de la cédula de identidad sean detectados.
# Se puede encontrar una buena explicación y pseudocódigo en los libros de [Burger & Burge](https://books.google.com.uy/books?id=YpzWCwAAQBAJ&printsec=frontcover#v=onepage&q&f=false). 

# In[ ]:


def HoughTransformLines(I_,tita_m,rho_n,thres):
    I = I_.copy()
    m = tita_m
    n = rho_n
    M, N = I.shape
    
    xr = M//2
    yr = N//2
    d_tita = np.pi/m
    d_radial = np.int(np.sqrt(M**2 + N**2)/n)
    v0 = n//2
    
    # Creo el acumulador
    A = np.zeros((m,n))
    
    # Lleno el acumulador
    for i in range(M):
        for j in range(N):
            if I[i,j] > 0:
                x = i-xr
                y = j-yr
                for u in range(m):
                    tita = d_tita * u
                    r = x*np.cos(tita) + y*np.sin(tita)
                    v = v0 + int(np.round(r/d_radial))
                    if v < n:
                        A[u,v] = A[u,v] + 1
    
    # Función para evaular máximo local
    def LocalMax(u,v):
        # Guardo el valor en una memoria
        aux = A[u,v]
        A[u,v] = 0
        # Sector del acumulador para evaluar
        A_ = A[np.max([0, u-1]):u+2, np.max([0,v-1]):v+2].copy()
        # Recupero el valor
        A[u,v] = aux
        return A[u,v] > np.max(A_)

    # Econtrar las celdas con valor más alto
    L = []
    for u in range(m):
        for v in range(n):
            if A[u,v] >= thres and LocalMax(u,v):
                tita = d_tita * u
                r = (v - v0)*d_radial
                a = A[u,v]
                l = np.array([tita, r, a])
                L.append(l)

    # Ordenamiento descendente de la lista
    L = np.array(L)
    L = np.array(sorted(L, key=lambda x : x[2], reverse=True))
    
    return L, A, (xr, yr)


# In[ ]:


def mostrarHough(img, A, L, centro, nroLineas=4, interseccion=False):
    M, N = img.shape
    xr, yr = centro
    
    # Ploteo de acumulador
    xticks = np.linspace(0, A.shape[1], 3, dtype=int)
    yticks = np.linspace(0, A.shape[0], 3, dtype=int)
    xlabels = ['-$r_{max}$', '0', '$r_{max}$']
    ylabels = ['0', '$\\frac{\pi}{2}$', '$\pi$']
    
    plt.figure(figsize=(15,12))
    ax1 = plt.subplot(121)
    ax1.imshow(255-A)
    ax1.set_title('Acumulador')
    ax1.set_xlabel('Rho  [$\sqrt{M^2 + N^2}/n$]')
    ax1.set_ylabel('Tita [$\pi / m$]')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels)
    # Ploteo de imagen + líneas de Hough
    plt.subplot(122)
    plt.imshow(img)
    for tita, r, a in L[:nroLineas]:
        
        a = np.cos(tita)
        b = np.sin(tita)
        x0 = (a * r) + xr
        y0 = (b * r) + yr

        if np.tan(tita) == 0:
            y_ini = 0
            y_fin = N
            x_ini = x_fin = y0
        else:
            y_ini = y0 - (1/np.tan(tita))*(-x0)
            y_fin = y0 - (1/np.tan(tita))*(N-x0)
            x_ini = 0
            x_fin = N

        plt.plot([y_ini,y_fin], [x_ini, x_fin],'r')

    plt.xlim((0, N))
    plt.ylim((M, 0))
    plt.title('Cédula con líneas de Hough detectadas')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Calcular puntos de corte
    puntos_interseccion = []
    
    if interseccion:
        # ordeno con respecto a los ángulos
        L_TitaSorted = np.array(sorted(L[:nroLineas], key=lambda x : x[0], reverse=True))
        titas = L_TitaSorted[:,0]
        rhos = L_TitaSorted[:,1]

        # Creo la matriz de cos y sen
        A_02 = np.array([[np.cos(titas[0]), np.sin(titas[0])],
                         [np.cos(titas[2]), np.sin(titas[2])]])

        A_03 = np.array([[np.cos(titas[0]), np.sin(titas[0])],
                         [np.cos(titas[3]), np.sin(titas[3])]])

        A_12 = np.array([[np.cos(titas[1]), np.sin(titas[1])],
                         [np.cos(titas[2]), np.sin(titas[2])]])

        A_13 = np.array([[np.cos(titas[1]), np.sin(titas[1])],
                         [np.cos(titas[3]), np.sin(titas[3])]])
        # Vectores de rho
        Rho_02 = np.array([rhos[0], rhos[2]])
        Rho_03 = np.array([rhos[0], rhos[3]])
        Rho_12 = np.array([rhos[1], rhos[2]])
        Rho_13 = np.array([rhos[1], rhos[3]])
        
        # Hallo los puntos de interseccion
        despl = np.array([xr, yr])
        puntos_interseccion.append(np.linalg.solve(A_02, Rho_02) + despl)
        puntos_interseccion.append(np.linalg.solve(A_03, Rho_03) + despl)
        puntos_interseccion.append(np.linalg.solve(A_12, Rho_12) + despl)
        puntos_interseccion.append(np.linalg.solve(A_13, Rho_13) + despl)
        
        puntos_interseccion = np.array(puntos_interseccion)
        puntos_complex = puntos_interseccion[:,0] + 1j*puntos_interseccion[:,1]
        puntos_ordenados = np.zeros_like(puntos_interseccion)
        puntos_ordenados[0,:] = puntos_interseccion[np.argmin(np.abs(puntos_complex))] 
        puntos_ordenados[1,:] = puntos_interseccion[np.argmin(np.abs(puntos_complex - np.array([0 + 1j*N])))]
        puntos_ordenados[2,:] = puntos_interseccion[np.argmin(np.abs(puntos_complex - np.array([M + 1j*N])))]
        puntos_ordenados[3,:] = puntos_interseccion[np.argmin(np.abs(puntos_complex - np.array([M + 1j*0])))]
        
        return puntos_ordenados


# In[ ]:


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


def filtroSobel(I):
    
    M, N = I.shape
    Gx = np.zeros((M, N))
    Gy = np.zeros((M, N))
    
    gradGx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gradGy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Gx = np.array([[np.sum(gradGx * I[i-1:i+2, j-1:j+2]) for j in range(1,N-1)] for i in range(1, M-1)])
    Gy = np.array([[np.sum(gradGy * I[i-1:i+2, j-1:j+2]) for j in range(1,N-1)] for i in range(1, M-1)])
    
    G = np.sqrt(Gx**2 + Gy**2).astype('uint8')
    
    return G


# In[ ]:


def BinarizarImagen(img, thresh=130):
    img_ = img.copy()
    img_[img_<thresh] = 0
    img_[img_>=thresh] = 1
    
    return img_


# c) Definir una resolución en el espacio de parámetros (Δθ y Δρ) y el umbral sobre el número de votos, que permita detectar adecuadamente los bordes de la cédula en la foto tomada.

# In[ ]:


t_ini = time.time()

# Filtro de detección de bordes Prewitt
cedulaPrewitt = filtroPrewitt(cedula)
cedula2Prewitt = filtroSobel(cedula2)

cedulaNoisePrewitt = filtroPrewitt(cedulaNoise)
cedula2NoisePrewitt = filtroPrewitt(cedula2Noise)

## IMAGEN DE PRUEBA
pruebaPrewitt = filtroPrewitt(imgPrueba)

tiempoEjecucion(t_ini)


# In[ ]:


# Filtrado de detección de bordes
cedulaPrewittBin = BinarizarImagen(cedulaPrewitt, 100)
cedula2PrewittBin = BinarizarImagen(cedula2Prewitt, 100)
cedulaNoisePrewittBin = BinarizarImagen(cedulaNoisePrewitt, 150)
cedula2NoisePrewittBin = BinarizarImagen(cedula2NoisePrewitt, 150)
pruebaPrewittBin = BinarizarImagen(pruebaPrewitt, 100)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(cedulaPrewitt)
plt.title('Cédula 1 con filtro Prewitt')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedulaPrewittBin)
plt.title('Cédula 1 con filtro Prewitt umbralizado')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(cedula2Prewitt)
plt.title('Cédula 2 con filtro Prewitt')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedula2PrewittBin)
plt.title('Cédula 2 con filtro Prewitt umbralizado')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(cedulaNoisePrewitt)
plt.title('Cédula 1 con filtro Prewitt')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedulaNoisePrewittBin)
plt.title('Cédula 1 con filtro Prewitt umbralizado')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(cedula2NoisePrewitt)
plt.title('Cédula 2 con filtro Prewitt')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedula2NoisePrewittBin)
plt.title('Cédula 2 con filtro Prewitt umbralizado')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:


t_ini = time.time()

res_tita = 250
res_rho = 400
threshold = 250

L_cedula1, A_cedula1, centro_cedula1 = HoughTransformLines(cedulaPrewittBin, res_tita, res_rho, threshold)
L_cedula2, A_cedula2, centro_cedula2 = HoughTransformLines(cedula2PrewittBin, res_tita, res_rho, threshold)
L_cedulaNoise1, A_cedulaNoise1, centro_cedulaNoise1 = HoughTransformLines(cedulaNoisePrewittBin,
                                                                         res_tita, res_rho, threshold)
L_cedulaNoise2, A_cedulaNoise2, centro_cedulaNoise2 = HoughTransformLines(cedula2NoisePrewittBin,
                                                                         res_tita, res_rho, threshold)

tiempoEjecucion(t_ini)


# In[ ]:


mostrarHough(cedula2PrewittBin, A_cedula2, L_cedula2, centro_cedula2, nroLineas=12)
mostrarHough(cedulaPrewittBin, A_cedula1, L_cedula1, centro_cedula1)


# In[ ]:


mostrarHough(cedula2NoisePrewittBin, A_cedulaNoise2, L_cedulaNoise2, centro_cedulaNoise2)
mostrarHough(cedulaNoisePrewittBin, A_cedulaNoise1, L_cedulaNoise1, centro_cedulaNoise1)


# c) Aplicar alguno de los métodos de reducción de ruido evaluados en la parte 1 para mejorar los resultados de la detección.

# In[ ]:


# sigma_cedula1 = estimate_sigma(cedulaNoise)
# sigma_cedula2 = estimate_sigma(cedula2Noise)

# cedulaDenoise = denoise_nl_means(cedulaNoise.astype(float), h=.7*sigma_cedula1, sigma=30, fast_mode=True)
# cedula2Denoise = denoise_nl_means(cedula2Noise.astype(float), h=.7*sigma_cedula2, sigma=30, fast_mode=True)

cedulaDenoise = meanFilter(cedulaNoise, filterSize=(5,5))
cedula2Denoise = meanFilter(cedula2Noise, filterSize=(5,5))


# In[ ]:


cedulaDenoisePrewitt = filtroPrewitt(cedulaDenoise)
cedula2DenoisePrewitt = filtroPrewitt(cedula2Denoise)


# In[ ]:


cedulaDenoisePrewittBin = BinarizarImagen(cedulaDenoisePrewitt, 100)
cedula2DenoisePrewittBin = BinarizarImagen(cedula2DenoisePrewitt, 100)

plt.figure(figsize=(15,10))
plt.subplot(131)
plt.imshow(cedulaNoise)
plt.title('Cédula + ruido')
plt.axis('off')
plt.subplot(132)
plt.imshow(cedulaDenoise)
plt.title('Cédula con filtro m5')
plt.axis('off')
plt.subplot(133)
plt.imshow(cedulaDenoisePrewittBin)
plt.title('Cédula con filtro Prewitt umbralizado')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(131)
plt.imshow(cedula2Noise)
plt.title('Cédula + ruido')
plt.axis('off')
plt.subplot(132)
plt.imshow(cedula2Denoise)
plt.title('Cédula con filtro m5')
plt.axis('off')
plt.subplot(133)
plt.imshow(cedula2DenoisePrewittBin)
plt.title('Cédula con filtro Prewitt umbralizado')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:


t_ini = time.time()

res_tita = 250
res_rho = 400
threshold = 250

L_denoise1, A_denoise1, centro_denoise1 = HoughTransformLines(cedulaDenoisePrewittBin, 
                                                              res_tita, res_rho, threshold)
L_denoise2, A_denoise2, centro_denoise2 = HoughTransformLines(cedula2DenoisePrewittBin, 
                                                              res_tita, res_rho, threshold)

tiempoEjecucion(t_ini)


# In[ ]:


puntosDenoise = mostrarHough(cedulaDenoisePrewittBin, A_denoise1, L_denoise1,
                             centro_denoise1, interseccion=True)

puntosDenoise2 = mostrarHough(cedula2DenoisePrewittBin, A_denoise2, L_denoise2,
                              centro_denoise2, interseccion=True)


# e) Hallar y aplicar una transformación geométrica que rectifique la imagen y la lleve a una imagen completa de 640x480.

# In[ ]:


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


def calcularHomografia(puntosA, puntosB):
    
    H, mask = cv2.findHomography(puntosA, puntosB)
    
    return H


# Devuelve la imagen transformada a partir de la matriz T
# Se debe poder elegir entre interpolacion vecino y bilineal 
# Si el punto está fuera de la imagen devuelve el valor "color_de_fondo" que por defecto es negro
def transformar(img, T, tipo_interpolacion='vecino', color_de_fondo=0, extender_V=False, extender_H=False):
    
    inv_T = np.linalg.inv(T)
    I_salida = np.ones((480, 640)) * color_de_fondo
    
    if extender_V:
        I_salida = np.vstack((img,np.zeros_like(img)))
    if extender_H:
        I_salida = np.hstack((img,np.zeros_like(img)))
        
    M, N = I_salida.shape[0:2]
        
    error_infinito = False
    error_interpolacion = False
    
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


# In[ ]:


puntosDestino = np.array([[0,0], [0, 640], [480, 640], [480, 0]])

H1 = calcularHomografia(puntosDenoise, puntosDestino)
H2 = calcularHomografia(puntosDenoise2, puntosDestino)

cedulaRectificada = transformar(cedulaDenoise, H1, tipo_interpolacion='bilineal')
cedula2Rectificada = transformar(cedula2Denoise, H2, tipo_interpolacion='bilineal')


# In[ ]:


plt.figure(figsize=(18,10))
plt.subplot(121)
plt.imshow(cedulaDenoise)
for i in range(4):
    plt.plot(puntosDenoise[i,1].astype(int), puntosDenoise[i,0].astype(int), 'or')
plt.title('Imagen original + ruido con filtrado m5')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedulaRectificada)
plt.title('Imagen rectificada')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(18,10))
plt.subplot(121)
plt.imshow(cedula2Denoise)
for i in range(4):
    plt.plot(puntosDenoise2[i,1].astype(int), puntosDenoise2[i,0].astype(int), 'or')
plt.title('Imagen original + ruido con filtrado m5')
plt.axis('off')
plt.subplot(122)
plt.imshow(cedula2Rectificada)
plt.title('Imagen rectificada')
plt.axis('off')
plt.tight_layout()
plt.show()


# ### 4) Detección de regiones  

# a) Dada una imagen, realizar la binarización mediante el uso de un umbral calculado automáticamente a partir del histograma.   
# Implementar para esto el algoritmo de Otsu.     
# http://en.wikipedia.org/wiki/Otsu%27s_method
# 
# Imágenes sugeridas: globulosA, globulosB.

# In[ ]:


def calcularHistograma(I, nBins):
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

def ecualizarHistograma(I):
    
    I_ecualizada = I.copy()
    
    hist_acc = histograma_acumulado(calcularHistograma(I, 256)) * 255
    
    def evaluarPixel(p):
        return int(np.round(hist_acc[p]))
    
    for i in np.arange(256):
        I_ecualizada[I==i] = evaluarPixel(i)

    return I_ecualizada


# In[ ]:


def otsu(histograma):

    #sumas acumuladas, p es un vector de prob. acumuladas para cada i
    p = np.zeros_like(histograma)
    for i in range(len(histograma)):
        p[i] = p[i-1] + histograma[i]
    # medias acumuladas, m es un vector de medias acumuladas para cada i
    m = np.zeros_like(histograma)
    for i in range(len(histograma)):
        m[i] = m[i-1] + i*histograma[i]
    
    # Media global
    m_G = m[-1]
    
    # varianza global
    largo = len(histograma)
    Var_G = np.sum((np.arange(largo)-m_G)**2 * histograma)
    
    Var_inter = np.zeros_like(histograma)
    for i in range(len(histograma)):
        if ((m_G*p[i] - m[i])**2) == 0:
            Var_inter[i] = 0
        else:
            try:
                Var_inter[i] = ((m_G*p[i] - m[i])**2) / (p[i]*(1-p[i]))
            except ZeroDivisionError:
                Var_inter[i] = 0
    
    umbral = np.argmax(Var_inter)
    
    return umbral


# In[ ]:


globulosB = pasarAGris(imageio.imread('./imagenes/globulosB.gif'))

hist_globulosB = calcularHistograma(globulosB, 256)
thresh = otsu(hist_globulosB)
globulosB_bin = BinarizarImagen(globulosB, thresh)


plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(globulosB)
plt.title('Imagen original')
plt.axis('off')
plt.subplot(122)
plt.imshow(globulosB_bin)
plt.title('Imagen binarizada con Otsu')
plt.axis('off')
plt.tight_layout()
plt.show()


# b) Proponer operaciones morfológicas que mejoren el resultado de la segmentación de la imagen globulosB intentando acercarse a que cada célula corresponda a una región.

# In[ ]:


# Ecualizar la imagen antes
globulosB_equ = ecualizarHistograma(globulosB)
hist_equ = calcularHistograma(globulosB_equ, 256)
thresh_equ = otsu(hist_equ)
globulosB_bin_equ = BinarizarImagen(globulosB_equ,thresh_equ)


# In[ ]:


plt.figure(figsize=(16,13))
plt.subplot(221)
plt.imshow(globulosB)
plt.title('Imagen original')
plt.axis('off')
plt.subplot(223)
plt.imshow(globulosB_bin)
plt.title('Imagen binarizada con Otsu')
plt.axis('off')
plt.subplot(222)
plt.imshow(globulosB_equ)
plt.title('Imagen ecualizada')
plt.axis('off')
plt.subplot(224)
plt.imshow(globulosB_bin_equ)
plt.title('Imagen ecualizada binarizada con Otsu')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:


# Elemento estructural conectividad 4
conect4 = sp.ndimage.generate_binary_structure(2, 1)
# Elemento estructural conectividad 8
conect8 = sp.ndimage.generate_binary_structure(2, 2)


# In[ ]:


inicial_bordes = np.zeros(globulosB_bin_equ.shape)
inicial_bordes[:,0] = inicial_bordes[:,-1] = inicial_bordes[0,:] = inicial_bordes[-1,:] = 1

## el parametro mask hace que solamente se modifiquen los pixeles de la imagen donde la mask vale 1
globulosB_rellenada = sp.ndimage.binary_dilation(inicial_bordes, structure=conect4, iterations=-1,
                                              mask=globulosB_bin_equ)


# In[ ]:


plt.figure(figsize=(14,6))
plt.subplot(121)
plt.imshow(globulosB_bin_equ)
plt.title('Imagen binarizada')
plt.axis('off')
plt.subplot(122)
plt.imshow(globulosB_rellenada)
plt.title('Imagen rellenada')
plt.axis('off')
plt.tight_layout()
plt.show()


# c) Implementar el etiquetado de regiones de una imagen binaria. Verificar el correcto 
# funcionamiento con la imagen “para_etiquetar.bmp” (tiene 4 regiones negras con conectividad 8 vecinos sobre fondo blanco).

# In[ ]:


def etiquetar(I, conectividad=4):
    
    M, N = I.shape
    I_etiquetada = np.zeros_like(I)
    
    N_label = 0
    lista_equivalencias = []
    
    for i in range(M):
        for j in range(N):
            if i > 0 and j > 0:
                A = I_etiquetada[i-1,j-1]
            else:
                A = 0
            if i > 0:
                B = I_etiquetada[i-1,j]
            else:
                B = 0
            if j > 0:
                C = I_etiquetada[i,j-1]
            else:
                C = 0
                
            if I[i,j] != 0:
                if A > 0:
                    I_etiquetada[i,j] = A
                elif B > 0 and C == 0:
                    I_etiquetada[i,j] = B
                elif B == 0 and C > 0:
                    I_etiquetada[i,j] = C
                elif B > 0 and C > 0 and B != C:
                    I_etiquetada[i,j] = C
                    if [B, C] not in lista_equivalencias:
                        lista_equivalencias.append([B,C])             
                else:
                    N_label += 1
                    I_etiquetada[i,j] = N_label
                    
            # elif del if I[j,m] != 0
            elif B > 0 and C > 0 and B != C:
                if [B, C] not in lista_equivalencias:
                    lista_equivalencias.append([B,C])
    
    # Genero listas de etiquetas sin repetir
    etiquetas_set = []
    while len(lista_equivalencias)>0:
        first, rest = lista_equivalencias[0], lista_equivalencias[1:]
        first = set(first)
        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
        etiquetas_set.append(first)
        lista_equivalencias = rest
    
    # Paso los sets a listas
    for item, i in zip(etiquetas_set, range(len(etiquetas_set))):
        etiquetas_set[i] = list(item)
        
    etiquetas_nuevas = np.array([equivalencia[0] for equivalencia in etiquetas_set])
    
    # Establezco etiquetas nuevas
    for equivalencia in etiquetas_set:
        indices = np.array([[I_etiquetada[i,j] in equivalencia for j in range(N)] for i in range(M)])
        I_etiquetada[indices] = equivalencia[0]
        
    print('Regiones encontradas: ', len(etiquetas_set))
    
    return I_etiquetada, etiquetas_nuevas


# In[ ]:


def invertirBinaria(img):
    return 1 - img


# In[ ]:


para_etiquetar = pasarAGris(imageio.imread('./imagenes/para_etiquetar.bmp'))
para_etiquetar_bin = BinarizarImagen(para_etiquetar, 128)
para_etiquetar_neg = invertirBinaria(para_etiquetar_bin)
img_etiquetada, img_etiquetas= etiquetar(para_etiquetar_neg)
img_etiquetada = 255 - img_etiquetada

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(para_etiquetar_bin)
plt.title('Imagen a etiquetar')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_etiquetada)
plt.title('Regiones de la imagen detectadas')
plt.axis('off')
plt.tight_layout()
plt.show()


# c) Etiquetar el resultado de la binarización realizada en la parte a) de la imagen: globulosB. Evaluar el resultado comentar los casos que no se pudieron resolver. correctamente

# In[ ]:


from skimage.color import label2rgb
label_colors = ('red', 'blue', 'yellow', 'magenta', 'green', 
                   'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen', 'lime',
               'maroon', 'gold') 


# In[ ]:


globulosB_rellenada_neg = invertirBinaria(globulosB_rellenada)
globulosB_etiquetados, etiquetas = etiquetar(globulosB_rellenada_neg)
globulosB_colores = label2rgb(globulosB_etiquetados,
                             bg_label=0, bg_color=(1,1,1), colors=label_colors)


# In[ ]:


from skimage.color import label2rgb

plt.figure(figsize=(15,6))
plt.subplot(121)
plt.imshow(globulosB_rellenada)
plt.title('Imagen a etiquetar')
plt.axis('off')
plt.subplot(122)
plt.imshow(globulosB_colores)
plt.title('Regiones de la imagen etiquetadas')
plt.axis('off')
plt.tight_layout()
plt.show()

