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
        prom=np.average(v)
        var=np.var(v)
        M[:,j]=(v-prom)/np.sqrt(var)
    return M

Vn=V_norm(np.copy(V))
print np.average(V[:,0])
print np.average(Vn[:,0])
print np.var(V[:,0])
print np.var(Vn[:,0])


#def m_cov(dT1,dT2,dT3,dT4):
#   var=np.array(([dT1],[dT2],[dT3],[dT4]))
#   M=np.zeros(shape=(4,4))
#   for j in range(4):
#      for i in range(4):
#         M[i,j]=sigma_ij(var[i,:],var[j,:])
#   return M


#print m_cov(dT1,dT2,dT3,dT4)
