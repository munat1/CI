import numpy as np
import scipy.cluster.vq as spcvq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

debug = True

######################################################################################
# Some function definition for easy access

def c1r(u): # computes the Radius for a given u in Class 1
    return np.sqrt(np.square(c1x1(u)) + np.square(c1x2(u)))

def c1a(u): # computes the Angle for a given u in Class 1
    return np.arctan2(c1x1(u), c1x2(u))

def c2r(u):
    return np.sqrt(np.square(c2x1(u)) + np.square(c2x2(u)))

def c2a(u):
    return np.arctan2(c2x1(u), c2x2(u))

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

c1 = (list(map(c1r, range(1,201))), list(map(c1a, range(1,201)))) # Class 1 points
c2 = (list(map(c2r, range(1,201))), list(map(c2a, range(1,201)))) # Class 2 points

plt.scatter(c1[0], c1[1], label="Class 1")
plt.scatter(c2[0], c2[1], label="Class 2")
plt.legend()
plt.savefig("polar_data.png")
plt.show()

###################################################################################
# Variables to play around with, associated with the NN itself
units = 50
learningrate = 0.03
sigma = 1.5 # Sigmas around this size lead to best results, I guess this is due to to the fact that this way different instances from the classes can "cover" for each other without completely overshadowing other instances
initvariant = 3 # 1: Random, 2: Random class points, 3: k-means

##################################################################################
# Creating the NN:

# Generation of location of hidden units
if initvariant == 1:
    # Variant 1: Random points in [-12.5:12.5][-12.5:12.5]
    # DONT USE THIS, THIS HASN'T BEEN ADAPTED TO POLAR COORDINATES
    rbfweights = np.vectorize(lambda x: x*25 - 12.5)(np.random.rand(units, 2))

elif initvariant == 2:
    # Variant 2: picking random points in the classes as location of hidden units
    # It's not k-means but still gives us reasonably good result as class entities are clustered
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
    # It will sometimes generate some outliers but the output is still fine
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
def g(r,a,target=None):
    rs = [] # ||x-c_h||
    os = [] # f_{act}(||x-c_h||)
    wos = [] # w_{h,\Sigma} * f_{act}(||x-c_h||)
    for i in range(0, units):
        rs.append(np.sqrt(np.square(r - rbfweights[i][0]) + np.square(a - rbfweights[i][1])))
        os.append(gauss(rs[i], rbfsigmas[i]))
        wos.append(os[i] * outputweights[i])
    o = 0.2 * sum(wos) # o_{\Sigma} # We use a linear output activation function 
    if target is not None: # Training
        for i in range(0,units):
            outputweights[i] = outputweights[i] + learningrate * (target - o) * os[i] # see slides, the derivative of the linear output activation function is included in the learningrate (as it's only a constant factor)
    return sum(wos)

#################################################################################
# Learn it-times, using random points from the classes defined above

def learn(it=500000):
    for i in range(0,it):
        if i%1000 == 0:
            print(i)
        cl = np.random.randint(1,3)
        u = np.random.randint(1,201)
        if cl == 1:
            g(c1r(u),c1a(u),target=1)
        else:
            g(c2r(u),c2a(u),target=-1)

if debug:
    print("Start leanrning...")
learn(it=1000000)
if debug:
    print("Outputweights after training :", outputweights)


##################################################################################
# visualisation of NN output
resolution = 0.10 # how finegrained the map is, here 0.1 is ok (runtime-wise) as |R| < |X|, |A| << |Y|
Ro = np.arange(0,17.0001,resolution)
Ao = np.arange(-np.pi,np.pi,resolution)
R, A = np.meshgrid(Ro, Ao)

# This calculation might actually take a while with a too fine resolution
Z = np.zeros(R.shape) # Output with linear activation funktion
for i in range(0,Ao.shape[0]):
    for j in range(0,Ro.shape[0]):
        Z[i][j] = g(Ro[j],Ao[i])


# Zeroth figure: Original classes and position of hidden units
#fig = plt.figure()
plt.scatter(c1[0],c1[1], label="CLass 1")
plt.scatter(c2[0],c2[1], label="Class 2")
plt.scatter(rbfweights[:,0],rbfweights[:,1], label="RBF Neurons")
plt.legend()
plt.savefig("polar_rbfs.png")
plt.show()

# First figure: Neural network output with linear activation function
fig = plt.figure() 
d3 = fig.add_subplot(1,1,1, projection='3d')

surf = d3.plot_surface(R, A, Z, cmap=cm.coolwarm, rstride=1, cstride=1, label="NN Output") # Output of NN
pts = d3.scatter(rbfweights[:,0], rbfweights[:,1],outputweights, label="RBF Neurons") # Locations (and weights) of hidden units
c1s = d3.scatter(c1[0],c1[1],1.001, label="Class 1") # Class 1
c2s = d3.scatter(c2[0],c2[1],-1.001, label="Class 2") # Class 2
plt.savefig("polar_output.png")
plt.show() # Display

# Visualisation of NN output for cartesic coordinates
resolution = 0.25 # This WILL take a long time. Lower it to ~0.25 or something like that for reasonable performance
Xo = np.arange(-16, 16, resolution)
Yo = np.arange(-16, 16, resolution)
X, Y = np.meshgrid(Xo, Yo)

Z = np.zeros(X.shape)
for i in range(0, Yo.shape[0]):
    for j in range(0, Xo.shape[0]):
        x = Xo[j]
        y = Yo[i]
        Z[i][j] = g(np.sqrt(x**2 + y**2), np.arctan2(x, y))

fig = plt.figure() 
d3 = fig.add_subplot(1,1,1, projection='3d')
surf = d3.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=1, cstride=1, label="NN Output") # Output of NN
c1s = d3.scatter(np.vectorize(c1x1)(np.arange(1,201)), np.vectorize(c1x2)(np.arange(1,201)), 1.0001, label="Class 1")
c2s = d3.scatter(np.vectorize(c2x1)(np.arange(1,201)), np.vectorize(c2x2)(np.arange(1,201)), -1.0001, label="Class 2")
plt.savefig("polar_cart.png")
plt.show()

