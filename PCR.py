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

nombre_v=['mean radius','radius SE','worst radius', 'mean texture','texture SE','worst texture', 'mean perimeter','perimeter SE','worst perimeter', 'mean area','area SE','worst area', 'mean smoothness','smoothness SE','worst smoothness', 'mean compactness','compactness SE','worst compactness', 'mean concavity','concavity SE','worst concavity', 'mean concave points','concave points SE','worst concave points', 'mean symmetry','symmetry SE','worst symmetry', 'mean fractal dimension','fractal dimension SE','worst fractal dimension']


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

#Funcion que calcula la covarianza entre dos variables.
def cov_ij(V1,V2):
    x=np.empty(0)
    mV1=np.mean(V1)
    mV2=np.mean(V2)
    for k in range(V1.size):
        x=np.append(x,((V1[k]-mV1)*(V2[k]-mV2))/(V1.shape[0]-1))
    return np.sum(x)

#Funcion que construye la matriz de covarianza.
def m_cov(Mn):
    n=Mn.shape[1]
    C=np.zeros(shape=(n,n))
    for j in range(n):
        for i in range(n):
            C[i,j]=cov_ij(Mn[:,i],Mn[:,j])
    return C

######### Calculos ##########
Vn=V_norm(np.copy(V))
Mc=m_cov(Vn)

val,vec=np.linalg.eig(Mc)
valp=np.flip(np.sort(val))
p_importantes=[]
for i in range(vec.shape[1]):
    for j in range(vec.shape[1]):
        if valp[i]==val[j]:
            a=vec[:,j].reshape(vec.shape[0],1)
            print "-----------------------"
            print "Valor propio: ", val[j]
            print "Vector propio: "
            print a
            if i==0:
                PC1=a
                x=np.argsort(a,axis=0)
                p1=np.array([nombre_v[x[-1, 0]], nombre_v[x[-2, 0]], nombre_v[x[-3, 0]]])
            elif i==1:
                PC2=a
                x=np.argsort(a,axis=0)
                p2=np.array([nombre_v[x[-1, 0]], nombre_v[x[-2, 0]], nombre_v[x[-3, 0]]])
PC=np.hstack((PC1,PC2))
print 'Los parametros mas importantes son ', p1, ' para PC1 y ', p2, ' para PC2.'
