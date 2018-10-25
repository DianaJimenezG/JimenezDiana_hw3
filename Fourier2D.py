import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm


########## Datos ##########
#Almacena la imagen en un arreglo de numpy.
img = plt.imread('arbol.png').astype(float)


########## Funciones ##########
#Funcion que elimina el ruido de la imagen. Esto se hace con un filtro pasa bajas.
#Remplaza todos los puntos mayores a 10**3.5 por el promedio de sus vecinos.
#Le entra la transformada de Fourier de la imagen original.
def filtro(f):
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if f[i,j]>=10**3.5:
                f[i,j]=(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1])/4.0
    return f


########## Transformaciones ##########
#Realiza la transformada de Fourier de la imagen.
img_fft = fftpack.fft2(img)
#Genera la transformada de Fourier filtrada
img_filtro=filtro(np.copy(img_fft))
#Reconstruccion de la imagen filtrada.
imagen=fftpack.ifft2(img_filtro).real


########## Graficas ##########
g=plt.figure(1)
plt.imshow(np.abs(np.real(img_fft)), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de Fourier')
plt.show()
g.savefig('JimenezDiana_FT2D.pdf')

h=plt.figure(2)
plt.imshow(np.abs(np.real(img_filtro)), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de Fourier con Filtro')
plt.show()
h.savefig('JimenezDiana_FT2D_filtrada.pdf')

k=plt.figure(3)
plt.imshow(imagen, plt.cm.gray)
plt.title('Imagen filtrada')
k.savefig('JimenezDiana_Imagen_filtrada.pdf')
