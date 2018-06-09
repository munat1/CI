import numpy as np
from math import exp, sin, sqrt, cos, atan, degrees, radians
from matplotlib import pyplot as plt

class Net:
    
    hidden = 200
    
    inputs = []
    hiddenActivity = []
    hiddenOutput = []
    activity = []
    output = []
    
    weights1 = []
    dweights1 = []
    dw1 = []
    weights2 = []
    dweights2 = []
    dw2 = []
    
    def __init__(self, o):
        self.inputs = np.zeros((2,))
        self.hiddenActivity = np.zeros((self.hidden,))
        self.hiddenOutput = np.zeros((self.hidden,))
        self.activity = np.zeros((1,))
        self.output = np.zeros((1,))
        
        if o == 1:
            self.weights1 = (np.random.rand(self.hidden,2)) * 16
        else:
            self.weights1 = (np.random.rand(self.hidden,2) - 0.5) * 32
        self.dweights1 = np.zeros((self.hidden,2))
        self.dw1 = np.zeros((self.hidden,2))
        self.weights2 = (np.random.rand(1,self.hidden) - 0.5)
        self.dweights2 = np.zeros((1,self.hidden))
        self.dw2 = np.zeros((1,self.hidden))
        
    def calc(self, ins):
        self.inputs = ins

        self.hiddenActivity = ((self.weights1 - ins)**2).sum(axis=1)
        self.hiddenOutput = np.vectorize(self.gauss)(self.hiddenActivity)

        self.activity = self.weights2.dot(self.hiddenOutput)
        self.output = np.vectorize(self.sigmoid)(self.activity)
        
        return self.output
        
    def train(self, ins, y, n):
        self.calc(ins)
        
        error = 0.5 * (self.output - y)**2
        
        s = (self.output - y) * np.vectorize(self.sigmoid)(self.activity)

        self.dw2 += -n * np.atleast_2d(s).T.dot(np.atleast_2d(self.hiddenOutput))

        k = -n * (s.dot(self.weights2) * np.vectorize(self.dgauss)(self.hiddenActivity) * self.hiddenActivity * 2)
        self.dw1 += ((self.weights1 - ins).T * k).T
        
        return error
    
    def addDW(self, a, b):
        self.dweights1 = self.dw1 * a + self.dweights1 * b
        self.dweights2 = self.dw2 * a + self.dweights2 * b
        
        self.weights2 += self.dweights2
        #self.weights1 += self.dweights1
        
        self.dw1 = np.zeros((self.hidden,2))
        self.dw2 = np.zeros((1,self.hidden))

    def gauss(self, x):
        return exp(-(x**2))
    
    def dgauss(self, x):
        return -2 * x * self.gauss(x)
        
    def sigmoid(self, x):
        return 1/(1 + exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def linear(self, x):
        return x
    
    def dlinear(self, x):
        return 1

net = Net(0)
net2 = Net(1)

def klasse1x1(u):
    return 2 + sin(0.2 * u + 8) * sqrt(u + 10)

def klasse1x2(u):
    return -1 + cos(0.2 * u + 8) * sqrt(u + 10)

def klasse2x1(u):
    return 2 + sin(0.2 * u - 8) * sqrt(u + 10)

def klasse2x2(u):
    return -1 + cos(0.2 * u - 8) * sqrt(u + 10)

def getW(x1, x2):
    t = degrees(atan(x2/x1))
    if x1 < 0:
        t += 180
    if x2 < 0:
        t += 360
    return radians(t%360)
    

k1x1 = [klasse1x1(u + 1) for u in range(200)]
k1x2 = [klasse1x2(u + 1) for u in range(200)]

k1r = [sqrt(k1x1[u]**2 + k1x2[u]**2) for u in range(200)]
k1w = [getW(k1x1[u],k1x2[u]) for u in range(200)]

y1 = np.array([1])
k2x1 = [klasse2x1(u + 1) for u in range(200)]
k2x2 = [klasse2x2(u + 1) for u in range(200)]

k2r = [sqrt(k2x1[u]**2 + k2x2[u]**2) for u in range(200)]
k2w = [getW(k2x1[u],k2x2[u]) for u in range(200)]

y2 = np.array([0])

plt.plot(k1x1,k1x2)
plt.plot(k2x1,k2x2)

plt.show()

plt.plot(k1w, k1r)
plt.plot(k2w, k2r)

plt.show()

print("training net1")
for j in range(200):
    e = 0
    for i in range(200):
        e += net.train(np.array([k1x1[i], k1x2[i]]), y1, 1)
        e += net.train(np.array([k2x1[i], k2x2[i]]), y2, 1)
        
        if i%5 == 0:
            net.addDW(1, 0.5)
    e /= 400
    print(e)
    
m = 10
res = np.zeros((32*m,32*m,3))
for x1 in range(-16*m,16*m):
    for x2 in range(-16*m,16*m):
        res[x1+16*m,x2+16*m] += net.calc(np.array([x1/m,x2/m]))[0]
        
print("done")
plt.imshow(res)
plt.show()

print("training polar net")
for j in range(200):
    e = 0
    for i in range(200):
        e += net2.train(np.array([k1w[i], k1r[i]]), y1, 1)
        e += net2.train(np.array([k2w[i], k2r[i]]), y2, 1)
        
        if i%5 == 0:
            net2.addDW(1, 0.5)
    e /= 400
    print(e)


m = 10
res = np.zeros((32*m,32*m,3))
for x1 in range(-16*m,16*m):
    if x1 == 0:
            continue
    for x2 in range(-16*m,16*m):
        w = getW(x1/m, x2/m)
        r = sqrt((x2/m)**2 + (x1/m)**2)

        res[x1+16*m,x2+16*m] += net2.calc(np.array([w,r]))[0]

print("done")
plt.imshow(res)
plt.show()