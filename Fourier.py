import numpy as np
import matplotlib.pyplot as plt

########## Datos ##########
datos=np.genfromtxt("signal.dat", delimiter=",")
t=datos[:,0]
y=datos[:,1]

#d_incompletos=np.genfromtxt("incompletos.dat", delimiter=",")
#t_inc=datos[:,0]
#y_inc=datos[:,1]

N=y.shape[0]

def transf_fourier(y, N):
    F=np.zeros(N)+0j
    for k in range(N):
        r=0.0
        i=0j
        for n in range(N):
            a=y[n]*np.exp(-1j*2.0*np.pi*k*(n/N))
            r+=np.real(a)
            i+=np.imag(a)
        F[k]=r+i
    return F

freq=np.fft.fftfreq(N, t[1]-t[0])
print transf_fourier(y, N)


plt.figure(1)
plt.plot(freq, np.real(transf_fourier(y, N)))
plt.show()
