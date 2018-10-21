import numpy as np
import imageio
from scipy import ndimage
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

img = ndimage.imread('tree.png')

print img.shape
print type(img)

########################################
#f = np.fft.fft2(img)
#fshift = np.fft.fftshift(f)
#magnitude_spectrum = 20*np.log(np.abs(fshift))

#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(f, cmap = 'gray')
#plt.title('Transformada'), plt.xticks([]), plt.yticks([])
#plt.show()
#######################################


im_fft = fftpack.fft2(img)

plt.figure()
plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Fourier transform')
