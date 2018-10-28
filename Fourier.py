import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, interpolate

########## Datos ##########
datos=np.genfromtxt("signal.dat", delimiter=",")
t=datos[:,0]
y=datos[:,1]

d_incompletos=np.genfromtxt("incompletos.dat", delimiter=",")
t_inc=d_incompletos[:,0]
y_inc=d_incompletos[:,1]

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

def filtro(fq, y, fc):
    for i in range(fq.size):
        if abs(fq[i])>=fc:
            y[i]=0.0
    return y


########## Calculos ##########
N=y.shape[0]
fou=transf_fourier(y, N)
freq=frecuencias(N, t[1]-t[0])
print "BONO: no se usa el paquete fftfreq para calcular las frecuencias."

#Implementacion de la funcion interp para calcular la interpolacion lineal, cuadratica y cubica.
ti=np.linspace(t_inc[0], t_inc[-1], 512)
y_cuad=interp(t_inc, y_inc, ti)[0]
y_cub=interp(t_inc, y_inc, ti)[1]
Ni=ti.size
freq_interp=frecuencias(Ni, ti[1]-ti[0])
fou_cuad=transf_fourier(y_cuad, Ni)
fou_cub=transf_fourier(y_cub, Ni)

y_filt=fftpack.ifft(filtro(freq, np.copy(fou), 1000)).real
y_cuad_filt=fftpack.ifft(filtro(freq, np.copy(fou_cuad), 1000)).real
y_cub_filt=fftpack.ifft(filtro(freq, np.copy(fou_cub), 1000)).real

y_filt5=fftpack.ifft(filtro(freq, np.copy(fou), 500)).real
y_cuad_filt5=fftpack.ifft(filtro(freq, np.copy(fou_cuad), 500)).real
y_cub_filt5=fftpack.ifft(filtro(freq, np.copy(fou_cub), 500)).real


########## Graficas ##########
g=plt.figure(1)
plt.plot(t,y)
plt.title('Senal original')
plt.ylabel('Amplitud')
plt.xlabel('Tiempo')
plt.show()
g.savefig('JimenezDiana_signal.pdf')

h=plt.figure(2)
plt.plot(freq, np.real(fou))
plt.title('Transformada de Fourier')
plt.ylabel('Transformada')
plt.xlabel('Frecuencia')
plt.xlim(-2000,2000)
plt.show()
h.savefig('JimenezDiana_TF.pdf')

o=plt.figure(3)
plt.plot(t,y_filt)
plt.title('Senal filtrada')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.show()
o.savefig('JimenezDiana_fitrada.pdf')

z=plt.figure(4)
plt.subplot(311)
plt.plot(freq, np.real(fou))
plt.xlim(-2000,2000)
plt.title('Transformada de Fourier')
plt.ylabel('Signal')
plt.subplot(312)
plt.plot(freq_interp, np.real(fou_cuad))
plt.xlim(-2000,2000)
plt.ylabel('Cuadratica')
plt.subplot(313)
plt.plot(freq_interp, np.real(fou_cub))
plt.xlim(-2000,2000)
plt.ylabel('Cubica')
plt.xlabel('Frecuencias')
plt.show()
z.savefig('JimenezDiana_TF_interpola.pdf')

u=plt.figure(5)
plt.subplot(211)
plt.plot(t, y_filt, label='Signal')
plt.plot(ti, y_cuad_filt, label='Cuadratica')
plt.plot(ti, y_cub_filt, label='Cubica')
plt.legend()
plt.title('Filtro de 1000Hz')
plt.ylabel('Amplitud')
plt.tick_params(labelbottom=False)
plt.subplot(212)
plt.plot(t, y_filt5, label='Signal')
plt.plot(ti, y_cuad_filt5, label='Cuadratica')
plt.plot(ti, y_cub_filt5, label='Cubica')
plt.legend()
plt.title('Filtro de 500Hz')
plt.ylabel('Amplitud')
plt.xlabel('Tiempo')
plt.show()
u.savefig('JimenezDiana_2Filtros.pdf')
