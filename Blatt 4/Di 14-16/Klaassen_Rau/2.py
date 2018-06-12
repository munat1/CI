# Copyright 2018 Malte Klaassen, Witali Rau
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.cluster.vq as spcvq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

debug = True

######################################################################################
# Some function definition for easy access

def c1x1(u):
    return 2 + np.sin(0.2 * u + 8) * np.sqrt(u + 10)
def c1x2(u):
    return -1 + np.cos(0.2 * u + 8) * np.sqrt(u + 10)

def c2x1(u):
    return 2 + np.sin(0.2 * u - 8) * np.sqrt(u + 10)
def c2x2(u):
    return -1 + np.cos(0.2 * u - 8) * np.sqrt(u + 10)

def gauss(r,sigma):
    return np.exp((- np.square(r))/(2*np.square(sigma)))

def fermi(x):
    return 1/(1+np.exp(-x))

####################################################################################

c1 = (list(map(c1x1, range(1,201))), list(map(c1x2, range(1,201)))) # Class 1 points
c2 = (list(map(c2x1, range(1,201))), list(map(c2x2, range(1,201)))) # Class 2 points

###################################################################################
# Variables to play around with, associated with the NN itself
units = 50
learningrate = 0.2
sigma = 2.5 # Sigmas around this size lead to best results, I guess this is due to to the fact that this way different instances from the classes can "cover" for each other without completely overshadowing other instances
initvariant = 3 # 1: Random, 2: Random class points, 3: k-means

##################################################################################
# Creating the NN:

# Generation of location of hidden units
if initvariant == 1:
    # Variant 1: Random points in [-12.5:12.5][-12.5:12.5]
    rbfweights = np.vectorize(lambda x: x*25 - 12.5)(np.random.rand(units, 2))

elif initvariant == 2:
    # Variant 2: picking random points in the classes as location of hidden units
    # It's not k-means but still gives us reasonably good result as class entities are clustered along a curved line
    rbfweights = np.zeros((units,2))
    c = list(zip(c1[0],c1[1])) + list(zip(c2[0],c2[1])) # Joint list of class points for alternative initial weight distribution
    for i in range(0,units):
        j = np.random.randint(0, len(c))
        rbfweights[i][0] = c[j][0]
        rbfweights[i][1] = c[j][1]
        c.pop(j)

else:
    # Variant 3: Using k-means
    # delivers pretty good results, we are using scipys implementation instead of making our own. 
    # at first glance the results of this are not significantly better than variant 2 but we should get a guarantee for proper distribution of the hidden units while variant 2 only provides this for a high numbers of hidden units
    # There seems to be an issue with this method as it will often lead to some outliers. Still generates decent data though ;)
    c = np.vstack(list(zip(c1[0],c1[1])) + list(zip(c2[0],c2[1])))
    centroid, label = spcvq.kmeans2(c, units)
    rbfweights = centroid

# Generation of sigmas for Gauss, we use constant sigmas
rbfsigmas = np.vectorize(lambda x: sigma)(np.zeros(units))

# Generation of initial weights for the output, randomly chosen in [-1,1]
outputweights = np.vectorize(lambda x: x*2 - 1)(np.random.rand(units))

if debug:
    print("Hidden Units: ", rbfweights)
    print("Sigmas :", rbfsigmas)
    print("Outputweights :", outputweights)

#################################################################################
# g calculates the result of the NN. If a target other than None is passed to this it will also train the network
def g(x1,x2,target=None):
    rs = [] # ||x-c_h||
    os = [] # f_{act}(||x-c_h||)
    wos = [] # w_{h,\Sigma} * f_{act}(||x-c_h||)
    for i in range(0, units):
        rs.append(np.sqrt(np.square(x1 - rbfweights[i][0]) + np.square(x2 - rbfweights[i][1])))
        os.append(gauss(rs[i], rbfsigmas[i]))
        wos.append(os[i] * outputweights[i])
    a = sum(wos) # o_{\Sigma} # We use a linear output activation function 
    o = a
    if target is not None: # Training
        for i in range(0,units):
            outputweights[i] = outputweights[i] + learningrate * (target - o) * os[i] # see slides, the derivative of the linear output activation function is included in the learningrate (as it's only a constant factor)
    return o

#################################################################################
# Learn it-times, using random points from the classes defined above

def learn(it=500000):
    for i in range(0,it):
        if i%1000 == 0:
            print(i)
        cl = np.random.randint(1,3)
        u = np.random.randint(1,201)
        if cl == 1:
            g(c1x1(u),c1x2(u),target=1)
        else:
            g(c2x1(u),c2x2(u),target=-1)

if debug:
    print("Start leanrning...")
learn(it=30000)
if debug:
    print("Outputweights after training :", outputweights)


##################################################################################
# visualisation of NN output
resolution = 0.25 # how fine the representation is - exercise said 0.1, I found this to take too much time and it made my laptop slow down to much, I found 0.25 to be a sweetspot
Xo = np.arange(-16,16.0001,resolution)
Yo = np.arange(-16,16.0001,resolution)
X, Y = np.meshgrid(Xo, Yo)

# This calculation might actually take a while with a too fine resolution
Z = np.zeros(X.shape) # Output with linear activation funktion
print(X.shape)
for i in range(0,Z.shape[0]):
    for j in range(0,Z.shape[1]):
        Z[j][i] = g(Xo[i],Yo[j])


# Zeroth figure: Original classes and position of hidden units
#fig = plt.figure()
plt.plot(c1[0],c1[1], label="Class 1")
plt.plot(c2[0],c2[1], label="Class 2")
plt.scatter(rbfweights[:,0],rbfweights[:,1], label="RBF Neurons")
plt.legend()
plt.savefig("cart_rbfs.png")
plt.show()

# First figure: Neural network output with linear activation function
fig = plt.figure() 
d3 = fig.add_subplot(1,1,1, projection='3d')

surf = d3.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=1, cstride=1) # Output of NN
pts = d3.scatter(rbfweights[:,0], rbfweights[:,1],outputweights) # Locations (and weights) of hidden units
c1s = d3.plot(c1[0],c1[1],1.001) # Class 1
c2s = d3.plot(c2[0],c2[1],-1.001) # Class 2
plt.savefig("cart_output.png")
plt.show() # Display

fig = plt.figure() 
d3 = fig.add_subplot(1,1,1, projection='3d')

surf = d3.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=1, cstride=1) # Output of NN
c1s = d3.plot(c1[0],c1[1],1.001) # Class 1
c2s = d3.plot(c2[0],c2[1],-1.001) # Class 2
plt.savefig("cart_output_2.png")
plt.show() # Display


