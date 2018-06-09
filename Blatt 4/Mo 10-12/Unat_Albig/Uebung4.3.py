import math
import numpy as np
import matplotlib.pyplot as plt

u = np.arange(1,201)

_x1 = [2+math.sin(0.2*t +8)*math.sqrt(t+10) for t in u]
_x2 = [-1+math.cos(0.2*t +8)*math.sqrt(t+10) for t in u]

_x4 = [-1+math.cos(0.2*t-8)*math.sqrt(t+10) for t in u]
_x3 = [2+math.sin(0.2*t -8)*math.sqrt(t+10) for t in u]

x1 = [math.sqrt(_x1[t]**2+_x2[t]**2) for t in range(200)]
x2 = [math.atan(_x2[t]/_x1[t]) for t in range(200)]

x3 = [math.sqrt(_x3[t]**2+_x4[t]**2) for t in range(200)]
x4= [math.atan(_x4[t]/_x3[t]) for t in range(200)]

class rbf:
    def __init__(self,z1,z2,variance):

        self.z1 = z1
        self.z2 = z2
        self.variance = variance
        self.output = 0

    def process(self,x1,x2):
        x1,x2 = x1,x2

        distance = math.sqrt((self.z1-x1)**2+(self.z2-x2)**2)
        output = math.exp((-distance**2)/(2*self.variance**2))
        self.output = output
        return output

neurons = []

r1 = np.random.randint(0,200,25)
r2 = np.random.randint(0,200,25)

variance = 0.2
z1_all=[]
z2_all=[]
for i in range(25):

    z1 = x1[r1[i]]
    z2 = x2[r1[i]]
    neurons.append(rbf(z1,z2,variance))
    z1_all.append(z1)
    z2_all.append(z2)
for i in range(25):

    z1 = x3[r2[i]]
    z2 = x4[r2[i]]
    neurons.append(rbf(z1,z2,variance))
    z1_all.append(z1)
    z2_all.append(z2)

weights = np.random.rand(50,1)*2-1
print(weights)


mu = 0.1

for i in range(200):
    output = sum(neurons[t].process(x1[i],x2[i])*weights[t] for t in range(50))
    for j in range(50):
        delta=mu*(1-output)*neurons[j].output
        weights[j] -=delta


for i in range(200):
    output = sum(neurons[t].process(x3[i],x4[i])*weights[t] for t in range(50))
    print(output)
    for j in range(50):
        delta = mu*(-1-output)*neurons[j].output
        weights[j] -=delta




o1 =[]
o2 =[]
for i in range(200):
    o1.append(1 if (sum(neurons[t].process(x1[i],x2[i])*weights[t] for t in range(50))>0) else -1)
for i in range(200):
    o2.append(1 if (sum(neurons[t].process(x3[i], x4[i]) * weights[t] for t in range(50)) > 0) else -1)

print(o1)
print(o2)
x1o = []
x2o = []
x3o = []
x4o = []
for i in range(200):
    if o1[i] == 1:
        x1o.append(x1[i])
        x2o.append(x2[i])
    else:
        x3o.append(x1[i])
        x4o.append(x2[i])

for i in range(200):
    if o2[i] == 1:
        x1o.append(x3[i])
        x2o.append(x4[i])
    else:
        x3o.append(x3[i])
        x4o.append(x4[i])


plt.scatter(x1o,x2o)
plt.scatter(x3o,x4o)
plt.scatter(z1_all,z2_all,color='black')

plt.plot(x1,x2,x3,x4)
plt.show()


