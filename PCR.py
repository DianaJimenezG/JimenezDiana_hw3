import numpy as np
import matplotlib.pyplot as plt

########## Datos ##########
#Matriz de variables
datos=np.genfromtxt("WDBC.dat", delimiter=",")
V=datos[:,2:]

#Columna de diagnostico. Se usa la convencion 'M'=1 y 'B'=0.
diagn=np.genfromtxt("WDBC.dat", delimiter = "," , usecols = 1, dtype = "str")
D=np.ones(diagn.shape)
for i in range(diagn.shape[0]):
    if diagn[i]=='B':
        D[i]=0

######### Funciones ##########
#Funcion que normalizacion la matriz de variables.
#Para cada columna, resta el promedio y divide por la varianza.
def V_norm(M):
    for j in range(M.shape[1]):
        v=M[:,j]
        p=np.average(v)
        var=np.var(v)
        M[:,j]=(v-p)/np.sqrt(var)
    return M

Vn=V_norm(np.copy(V))

def cov_ij(V1,V2):
    x=np.empty(0)
    mV1=np.mean(V1)
    mV2=np.mean(V2)
    for k in range(V1.size):
        x=np.append(x,((V1[k]-mV1)*(V2[k]-mV2))/(V1.shape[0]-1))
    return np.sum(x)

def m_cov(Mn):
    n=Mn.shape[1]
    C=np.zeros(shape=(n,n))
    for j in range(n):
        for i in range(n):
            C[i,j]=cov_ij(Mn[:,i],Mn[:,j])
    return C

Mc=m_cov(Vn)
m=np.cov(np.transpose(Vn))
print np.allclose(Mc,m)
