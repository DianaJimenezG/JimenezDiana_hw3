import numpy as np
import matplotlib.pyplot as plt

datos=np.genfromtxt("WDBC.dat", delimiter=",", dtype = "str")
V=datos[:,2:]

diagn=np.genfromtxt("WDBC.dat", delimiter = "," , usecols = 1, dtype = "str")
D=np.ones(diagn.shape)
for i in range(diagn.shape[0]):
    if diagn[i]=='B':
        D[i]=0

#g=plt.figure(1)
#plt.plot(T1)
#plt.plot(T2)
#plt.plot(T3)
#plt.plot(T4)
#plt.legend(['T1:fl','T2:fr','T3:bl','T4:br'], loc='upper left')
#plt.ylabel('Temperatura')
#g.savefig("JimenezDianaS6C2PLOTTemp.pdf")


#T1_prom=np.average(T1)
#T2_prom=np.average(T2)
#T3_prom=np.average(T3)
#T4_prom=np.average(T4)


#def x_xprom(x,xprom):
#   d=np.empty(0)
#   for i in range(x.size):
#      d=np.append(d, x[i]-xprom)
#   return d

#dT1=x_xprom(T1,T1_prom)
#dT2=x_xprom(T2,T2_prom)
#dT3=x_xprom(T3,T3_prom)
#dT4=x_xprom(T4,T4_prom)

#def sigma_ij(dTa,dTb):
#   x=np.empty(0)
#   for k in range(dTa.size-1):
#      x=np.append(x,(dTa[k]*dTb[k])/(dTa.size-1))
#   return np.sum(x)

#def m_cov(dT1,dT2,dT3,dT4):
#   var=np.array(([dT1],[dT2],[dT3],[dT4]))
#   M=np.zeros(shape=(4,4))
#   for j in range(4):
#      for i in range(4):
#         M[i,j]=sigma_ij(var[i,:],var[j,:])
#   return M


#print m_cov(dT1,dT2,dT3,dT4)
