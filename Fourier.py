import numpy as np
import matplotlib.pyplot as plt

########## Datos ##########
datos=np.genfromtxt("signal.dat", delimiter=",")
t=datos[:,0]
y=datos[:,1]

d_incompletos=np.genfromtxt("incompletos.dat", delimiter=",")
t_inc=datos[:,0]
y_inc=datos[:,1]

N=y.shape[0]
print N

def transf_fourier(y, N):
    F=np.zeros(N)
    for k in range(N):
        s=np.empty(0)
        for n in range(N):
            s=np.append(s,np.real(y[n]*np.exp(-1j*2.0*np.pi*k*(n/N))))
        F[k]=np.sum(s)
    return F

print transf_fourier(y, N)
freq=np.fft.fftfreq(N, t[1]-t[0])

print freq

plt.figure(1)
plt.plot(freq, transf_fourier(y, N))
plt.show()
