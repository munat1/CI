import random,time,sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)           #debugging

#aktivierungsfunktion
def activation(r_h,phi_h):
    return exp(-pow(r_h,2)/2*pow(phi_h,2))

#training data
u=np.linspace(1,200,200)            #u=1,...,200

K1_X1=2+np.sin(0.2*u+8)*np.sqrt(u+10)
K1_X2=-1+np.cos(0.2*u+8)*np.sqrt(u+10)
Y1=1

K2_X1=2+np.sin(0.2*u-8)*np.sqrt(u+10)
K2_X2=-1+np.cos(0.2*u-8)*np.sqrt(u+10)
Y2=-1

#Visualisierung der verschiedenen Datenpunkte
plt.plot(u,K1_X1,'bx')
plt.plot(u,K1_X2,'rx')
plt.plot(u,K2_X1,'gx')
plt.plot(u,K2_X2,'yx')
plt.xlabel('u')
plt.ylabel('f(u)')
plt.show()

#keine Zeit mehr für restliches Programm
