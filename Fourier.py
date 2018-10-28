import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

########## Datos ##########
datos=np.genfromtxt("signal.dat", delimiter=",")
t=datos[:,0]
y=datos[:,1]

d_incompletos=np.genfromtxt("incompletos.dat", delimiter=",")
t_inc=datos[:,0]
y_inc=datos[:,1]

########## Funciones ##########
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

#Funcion que calcula la funcion de interpolacion cuadratica y cubica con la funcion interp1d de scipy.
def interp(x, y, xi):
    cuad=interpolate.interp1d(x, y, kind='quadratic')
    cub=interpolate.interp1d(x, y, kind='cubic')
    y_cuad=cuad(xi)
    y_cub=cub(xi)
    return y_cuad, y_cub


########## Calculos ##########
N=y.shape[0]
fou=transf_fourier(y, N)
freq=frecuencias(N, t[1]-t[0])
print "BONO: no se usa el paquete fftfreq para calcular las frecuencias."

#Implementacion de la funcion interp para calcular la interpolacion lineal, cuadratica y cubica.
ti=np.linspace(t_inc[1], t_inc[-1], 512)
y_cuad=interp(t_inc, y_inc, ti)[0]
y_cub=interp(t_inc, y_inc, ti)[1]
Ni=ti.size
freq_interp=frecuencias(Ni, ti[1]-ti[0])
fou_cuad=transf_fourier(y_cuad, Ni)
fou_cub=transf_fourier(y_cub, Ni)


########## Graficas ##########
g=plt.figure(1)
plt.plot(t,y)
plt.title('Senal original')
plt.ylabel('y')
plt.xlabel('Tiempo')
plt.show()
g.savefig('JimenezDiana_signal.pdf')

h=plt.figure(2)
plt.plot(freq, np.abs(np.real(fou)))
plt.title('Transformada de Fourier')
plt.ylabel('Amplitud')
plt.xlabel('Frecuencia')
plt.show()
h.savefig('JimenezDiana_TF.pdf')
