import copy, numpy as np
#np.random.seed(0)           #debugging

#aktivierungsfunktion
def activation(x):
    return 1/(1+np.exp(-x))

#ableitung aktivierungsfunktion
def activ_deriv(output):
    return output*(1-output)

#training data
int2binary = {}
dimension = 8

largest = pow(2,dimension)
binary = np.unpackbits(np.array([range(largest)],dtype=np.uint8).T,axis=1)
for i in range(largest):
    int2binary[i] = binary[i]

#inputvariablen
eta = 0.1
input = 2
hidden = 16
output = 1


#gewichte
synapse_0 = 2*np.random.random((input,hidden)) - 1
synapse_1 = 2*np.random.random((hidden,output)) - 1
synapse_h = 2*np.random.random((hidden,hidden)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#training
for j in range(10000):
    a_int = np.random.randint(largest/2) #int-wert
    a = int2binary[a_int] #binärwert

    b_int = np.random.randint(largest/2) #int-wert
    b = int2binary[b_int] #binärwert

    #korrektes Ergebnis
    c_int = a_int + b_int
    c = int2binary[c_int]

    #beste Schätzung
    d = np.zeros_like(c)

    error = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden))

    for position in range(dimension):

        #input & output generieren
        X = np.array([[a[dimension - position - 1],b[dimension - position - 1]]])
        y = np.array([[c[dimension - position - 1]]]).T

        #hidden layer
        layer_1 = activation(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        #output layer
        layer_2 = activation(np.dot(layer_1,synapse_1))

        #error
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*activ_deriv(layer_2))
        error += np.abs(layer_2_error[0])

        #binär in dezimal umwandeln
        d[dimension - position - 1] = np.round(layer_2[0][0])

        #hidden layer für nächsten Schritt speichern
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden)

    for position in range(dimension):

        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        #error des outputs
        layer_2_delta = layer_2_deltas[-position-1]
        #error des hidden layers
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * activ_deriv(layer_1)

        #gewichtsänderung
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta


    synapse_0 += synapse_0_update * eta
    synapse_1 += synapse_1_update * eta
    synapse_h += synapse_h_update * eta

    synapse_0_update = 0
    synapse_1_update = 0
    synapse_h_update = 0

    if(j % 1000 == 0):
        print("Zahl 1:"+str(a))
        print("Zahl 2:"+str(b))
        print("Error:"+str(error))
        print("Pred:"+str(d))
        print("True:"+str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print("------------")
