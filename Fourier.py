import numpy as np
import matplotlib.pyplot as plt

########## Datos ##########
datos=np.genfromtxt("signal.dat", delimiter=",")
t=datos[:,0]
y=datos[:,1]

d_incompletos=np.genfromtxt("incompletos.dat", delimiter=",")
t_inc=datos[:,0]
y_inc=datos[:,1]

def transf_fourier(y, N):
    F = np.zeros(N) + 0j
    n = np.arange(N)
    for k in range(N):
            F[k] = np.sum(y*np.exp(-1j*2.0*np.pi*k*n/N))
    return F

def frecuencias(N, dt):
    if N%2.0 == 0:
        frec=np.concatenate((np.linspace(0, N/2.0-1, N/2.0), np.linspace(-N/2.0, -1, N/2.0)))/(dt*N)
    else:
        frec=np.concatenate((np.linspace(0, (N-1)/2.0, ((N-1)/2.0)+1), np.linspace(-(N-1)/2.0, -1, (N-1)/2.0)))/(dt*N)
    return frec

N=y.shape[0]
freq=frecuencias(N, t[1]-t[0])

plt.figure(1)
plt.plot(freq, np.real(transf_fourier(y, N)))
plt.show()
