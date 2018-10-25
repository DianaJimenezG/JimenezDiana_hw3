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
