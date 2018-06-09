import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm, pinv
from scipy import *

def eukl_dist(x, c):
	sum = 0
	for i, _ in enumerate(x):
		sum += (x[i] - c[i])**2
	return np.sqrt(sum)

def klasse1_x1(u):
	return 2 + math.sin( 0.2 * u + 8 ) * math.sqrt( u + 10 )
def klasse1_x2(u):
	return -1 + math.cos( 0.2 * u + 8 ) * math.sqrt( u + 10 )
def klasse1_y(u):
	return 1
	
def klasse2_x1(u):
	return 2 + math.sin( 0.2 * u - 8 ) * math.sqrt( u + 10 )
def klasse2_x2(u):
	return -1 + math.cos( 0.2 * u - 8 ) * math.sqrt( u + 10 )
def klasse2_y(u):
	return -1
	
def to_polar(x1, x2):
	r = np.sqrt(x1**2 + x2**2)
	t = np.arctan2(x2, x1)
	return (r, t)		

	
# Eingabe
us = np.linspace(1,200,200)
zero_pt = np.linspace(0,0,200)

k1_x1_us = np.vectorize(klasse1_x1)(us)
k1_x2_us = np.vectorize(klasse1_x2)(us)
k1_us = np.vectorize(klasse1_y)(us)

k2_x1_us = np.vectorize(klasse2_x1)(us)
k2_x2_us = np.vectorize(klasse2_x2)(us)
k2_us = np.vectorize(klasse2_y)(us)

# Polarkoordinaten: x1 = radius, x2 = winkel
# Kartesische Koordinaten zu Polarkoordinaten
#for i in range(len(k1_x1_us)):
#	k1_x1_us[i], k1_x2_us[i] = to_polar(k1_x1_us[i], k1_x2_us[i])
#	k2_x1_us[i], k2_x2_us[i] = to_polar(k2_x1_us[i], k2_x2_us[i])
#############################################

# NETZ
# 2 Eingabe Units (x1, x2)
# 50 Hidden Units
# 1 Ausgabe Unit (-1,1)
# Baue neuronales netz, was pro eingabe -> -1 oder 1 ausgibt, je nach klasse

# Trainiere

HIDDEN_NEURONS = 50
centers = [random.uniform(-1, 1, 2) for i in range(HIDDEN_NEURONS)]
weights = random.random((HIDDEN_NEURONS, 1))
		
def dist(v):
	global HIDDEN_NEURONS, centers
	d = zeros((v.shape[0], HIDDEN_NEURONS), float)
	for ci, c in enumerate(centers):
		for xi, x in enumerate(v):
			d[xi,ci] = eukl_dist( c, x )
	return d
	
def run(x):
	global weights
	d = dist(x)
	return dot(d, weights)

def train(x, y):
	global HIDDEN_NEURONS, centers, weights
	# Zufälliger index wird zum trainieren verwendet
	idx = random.permutation(x.shape[0])[:HIDDEN_NEURONS]
	centers = [x[i,:] for i in idx]
	d = dist(x)
	weights = dot(pinv(d), y)

	
k1_input = np.array(list(zip( k1_x1_us, k1_x2_us )))
k2_input = np.array(list(zip( k2_x1_us, k2_x2_us )))

# alles wird auf einmal trainiert
_input = np.array(list(zip( np.append( k1_x1_us, k2_x1_us ), np.append(k1_x2_us, k2_x2_us ) )))
_expected =  np.append( k1_us, k2_us )
train(_input, _expected)

# Visualisierung
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax = plt.axes(projection='3d')
ax.plot3D(k1_x1_us, k1_x2_us, zero_pt, 'red')
ax.plot3D(k2_x1_us, k2_x2_us, zero_pt, 'blue')

# Zentren visualisieren
for c in centers:
	ax.scatter(c[0],c[1],0.2, c='orange')	

# Punkte, die die ausgabe des netzes darstellen
ranges = np.arange(-16, 16, 1)#0.1) # 0.1er Schritte dauern zu lange, daher hier 0.5er bzw 1er Schritte
for i in ranges:
	for j in ranges:
		inp = np.array(( (i, j), ))
		
		result = run( inp )[0]
		
		if result > 0:
			ax.scatter(i,j,0, c='green')
		else:
			ax.scatter(i,j,0, c='yellow')

plt.show()

# Klasse 1 wird in grün dargestellt, Klasse 2 in gelb
# Die Neuronenzentren werden in Orange dargestellt

# Leider sieht die Ausgabe mit 50 Neuronen nicht sonderlich spektakulär aus.
# Wenn jedoch die Anzahl auf 500 erhöht wird, entsteht vermutlich ein Overfitting, 
# jedoch kommt dann das Spiralenmuster viel deutlicher zum Vorschein.
 

