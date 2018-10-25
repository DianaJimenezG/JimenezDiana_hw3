import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm

img = plt.imread('arbol.png').astype(float)

plt.figure(1)
plt.imshow(img, plt.cm.gray)
plt.title('Imagen original')
plt.show()

im_fft = fftpack.fft2(img)
im_fft_2 = fftpack.fftshift(im_fft)

plt.figure(2)
plt.imshow(np.abs(np.real(im_fft_2)), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de Fourier')
plt.show()

def filtro(f):
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if f[i,j]>=10**3.5:
                f[i,j]=(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1])/4.0
    return f

img_filt=filtro(im_fft)

plt.figure(3)
plt.imshow(np.abs(np.real(img_filt)), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Transformada de Fourier con Filtro')
plt.show()

imagen=fftpack.ifft2(img_filt).real

plt.figure(4)
plt.imshow(imagen, plt.cm.gray)
plt.title('Imagen filtrada')
plt.show()
