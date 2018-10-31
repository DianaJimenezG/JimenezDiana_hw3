import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fftpack
from matplotlib.colors import LogNorm

########## Datos ##########
#Almacena la imagen en un arreglo de numpy.
img = plt.imread('arbol.png').astype(float)


########## Funciones ##########
#Funcion que elimina el ruido de la imagen. Esto se hace con un filtro pasa bajas.
#Remplaza por el promedio de sus vecinos todos los puntos mayores al promedio de los 25 puntos del centro.
#Le entra la transformada de Fourier con shift de la imagen original.
def filtro(f):
    max=f[0,0]
    n=1
    for k in range(-5,5):
        for g in range(-5,5):
            max=max+f[f.shape[0]/2+k,f.shape[1]/2+g]
            n=n+1
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if f[i,j]>=max/n and (i>=f.shape[0]/2+5 or i<=f.shape[0]/2-5) and (j>=f.shape[1]/2+5 or j<=f.shape[1]/2-5):
                f[i,j]=(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1])/4.0
    return f


########## Transformaciones ##########
#Realiza la transformada de Fourier de la imagen y le aplica shift.
img_fft = fftpack.fftshift(fftpack.fft2(img))
#Genera la transformada de Fourier filtrada
img_filtro=filtro(np.copy(img_fft))
#Reconstruccion de la imagen filtrada.
imagen=fftpack.ifft2(fftpack.ifftshift(img_filtro)).real


########## Graficas ##########
g=plt.figure(1)
plt.imshow(np.abs(np.real(img_fft)), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de Fourier')
plt.xlabel('w1')
plt.ylabel('w2')
g.savefig('JimenezDiana_FT2D.pdf')

h=plt.figure(2)
plt.imshow(np.abs(np.real(img_filtro)), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de Fourier con Filtro')
plt.xlabel('w1')
plt.ylabel('w2')
h.savefig('JimenezDiana_FT2D_filtrada.pdf')

k=plt.figure(3)
plt.imshow(imagen, plt.cm.gray)
plt.title('Imagen filtrada')
plt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
k.savefig('JimenezDiana_Imagen_filtrada.pdf')
