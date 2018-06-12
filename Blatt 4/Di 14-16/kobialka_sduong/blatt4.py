import math as m
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from scipy.spatial import distance
from random import shuffle
from numpy.core.defchararray import center


def fermifunktion(x):
    return 1-2*(1/(np.exp(-x)+1))

def ableitung(x):
    return (2*m.exp(-x))/(m.exp(-x)+1)**2

#karthesisch koordinaten
def klasse1(u):
    y=[]
    z=[]
    y1=(2+m.sin(0.2*u+8)*m.sqrt((u+10)))
    y2=(-1+m.cos(0.2*u+8)*m.sqrt((u+10)))
    y.append(y1)
    y.append(y2)
    z.append(y)
    z.append(-1)
    return z
    
def klasse2(u):
    y=[]
    z=[]
    y1=2+m.sin(0.2*u-8)*m.sqrt((u+10))
    y2=-1+m.cos(0.2*u-8)*m.sqrt((u+10))
    y.append(y1)
    y.append(y2)
    z.append(y)
    z.append(1)
    return z

#umrechnung Polarkoordianten
def cart2pol(x):
    r=np.sqrt(x[0]**2+x[1]**2)
    p=np.arctan2(x[1],x[0])
    return [r,p]

def pol2cart(x):
    x1=x[0]*np.cos(x[1])
    x2=x[0]*np.sin(x[1])
    return [x1,x2]

K1=[]       #klasse 1/klasse2 in form [(x1,x2),kontrollwert]
K2=[] 
K1x1=[]
K1x2=[]
K2x1=[]
K2x2=[]
KG=[]
#datenpunkte
for i in range(1,201):
    y1=klasse1(i)
    y2=klasse2(i)
    K1.append(y1)
    K2.append(y2)
    K1x1.append((y1[0])[0])
    K1x2.append((y1[0])[1])
    K2x1.append((y2[0])[0])
    K2x2.append((y2[0])[1])
    KG.append(y1)
    KG.append(y2)
        
fig=plt.figure()
sub1=fig.add_subplot(221)
sub1.scatter(K1x1,K1x2)
sub1.scatter(K2x1,K2x2)
sub1.set_title('datenpunkte karthesisch')


#globale Variablen
sigma=0
centerArray=[]
ausgabeNeuronenschicht=[]
ausgabeGewichte=np.random.rand(50)-0.5


def activationGauss(r):
    global sigma
    r=(-np.square(r))
    r=np.array(r)
    r=r/(2*np.square(sigma))
    return np.exp(r)

def computeCenter(array1,array2):
    global centerArray
    for i in range(0,25):
        r1=randint(0,199)
        r2=randint(0,199)
        centerArray.append(array1[r1][0])
        centerArray.append(array2[r2][0])
    return centerArray

def computeSigma():
    global centerArray
    summe=0
    for i in range(len(centerArray)):
        sigma=0
        for j in range(len(centerArray)):
            dist=distance.euclidean(centerArray[i], centerArray[j])
            sigma=sigma+dist
            summe=summe+1
    return sigma/summe
summe=1
centerArray=computeCenter(K1, K2)
sigma=computeSigma()

def ausgabe(x):
    global ausgabeNeuronenschicht,sigma, centerArray
    ausgabeArray=[]
    for i in range(len(centerArray)):
        d=distance.euclidean(x,centerArray[i])
        ausgabeArray.append(d)
    ausgabeArray=np.array(ausgabeArray)
    ausgabeNeuronenschicht=activationGauss(ausgabeArray)
    summe=np.sum(ausgabeNeuronenschicht)
    y=np.dot(ausgabeNeuronenschicht,ausgabeGewichte)
    y=fermifunktion(y/summe)
    return y

def ausgabePol(x):
    global ausgabeNeuronenschicht,sigma, centerArray
    ausgabeArray=[]
    for i in range(len(centerArray)):
        d=distance.euclidean(x,centerArray[i])
        ausgabeArray.append(d)
    ausgabeArray=np.array(ausgabeArray)
    ausgabeNeuronenschicht=activationGauss(ausgabeArray)
    y=np.dot(ausgabeNeuronenschicht,ausgabeGewichte)
    return y

def training(x,f):
    global ausgabeNeuronenschicht,ausgabeGewichte,centerArray
    n=0.001
    y=ausgabe(x)
    delta=n*(f-y)*ausgabeNeuronenschicht*ableitung(sum(ausgabeNeuronenschicht))
    ausgabeGewichte=ausgabeGewichte+delta

def trainingPol(x,f):
    global ausgabeNeuronenschicht,ausgabeGewichte,centerArray
    n=0.001
    y=ausgabePol(x)
    delta=n*(f-y)*ausgabeNeuronenschicht
    ausgabeGewichte=ausgabeGewichte+delta


'aufgabe 2'
A1=[]
A2=[]
for i in range(len(KG)):
    y=ausgabe((KG[i])[0])
    x1=((KG[i])[0])[0]
    x2=((KG[i])[0])[1]
    if(y>=0.0):
        A1.append(y)
    if(y<0.0):
        A2.append(y)
 

shuffle(KG)     
for j in range(20):
    for i in range(len(KG)):
        x=KG[i][0]
        f=KG[i][1]
        training(x,f)
    
x1=np.arange(-16,16.01,0.5)
x2=np.arange(-16,16.01,0.5)
x11=[]
x12=[]
x22=[]
x21=[]
for i in x1:
    for j in x2:
        x=[i,j]
        y=ausgabePol(x)
        if(y>0):
            x11.append(i)
            x12.append(j)
        else:
            x21.append(i)
            x22.append(j)
            
   
y1=[]
y2=[]

for i in range(len(centerArray)):
    y1.append(centerArray[i][0])
    y2.append(centerArray[i][1])
     

sub2=fig.add_subplot(222)
sub2.scatter(x11,x12,c='b',marker="o")
sub2.scatter(x21,x22,c='sandybrown',marker="o")
sub2.scatter(y1,y2,c='m',marker="P")
sub2.set_title('ausgabe karthesisch')


'aufgabe 3'
ausgabeGewichte=[]
ausgabeGewichte=np.random.rand(50)-0.5


KGp=[]
K1p=[]
K2p=[]
pR1=[]
pP1=[]
pR2=[]
pP2=[]
for i in range(len(K1)):
    x1=K1[i][0]
    x2=K2[i][0]
    f1=K1[i][1]
    f1=K2[i][1]
    x1=cart2pol(x1)
    x2=cart2pol(x2)
    K1p.append(x1)
    K2p.append(x2)
    pR1.append(x1[0])
    pP1.append(x1[1])
    pR2.append(x2[0])
    pP2.append(x2[1])
    KGp.append([x1,f])
    KGp.append([x2,f])

sub3=fig.add_subplot(2,2,3)
sub3.scatter(pR1,pP1)
sub3.scatter(pR2,pP2)
sub3.set_title('daten polar')

centerArray=[]
centerArray=computeCenter(K1p, K2p)
shuffle(KGp)     
for j in range(20):
    for i in range(len(KGp)):
        x=KGp[i][0]
        f=KGp[i][1]
        training(x,f)
    
x1=np.arange(0,16.01,0.5)
x2=np.arange(-3,3.01,0.1)
x11p=[]
x12p=[]
x22p=[]
x21p=[]
for i in x1:
    for j in x2:
        x=[i,j]
        y=ausgabe(x)
        if(y>0):
            x11p.append(i)
            x12p.append(j)
        else:
            x21p.append(i)
            x22p.append(j)
            
   
y1=[]
y2=[]
for i in range(len(centerArray)):
    y1.append(centerArray[i][0])
    y2.append(centerArray[i][1])
     
sub4=fig.add_subplot(2,2,4)
sub4.scatter(x11p,x12p)
sub4.scatter(x21p,x22p)
sub2.scatter(y1,y2,c='m',marker="P")
sub4.set_title('ausgabe polar')

    
plt.show()