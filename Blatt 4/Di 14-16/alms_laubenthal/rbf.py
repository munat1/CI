import numpy as np
import matplotlib.pyplot as plt

global fact


u = np.linspace(-16, 16, 321)
u = np.arange(200) + 1

k1 = np.zeros((200, 2))
k1[:,0] =  2 + np.sin(0.2 * u + 8) * np.sqrt(u + 10)
k1[:,1] = -1 + np.cos(0.2 * u + 8) * np.sqrt(u + 10)

k2 = np.zeros((200, 2))
k2[:,0] =  2 + np.sin(0.2 * u - 8) * np.sqrt(u + 10)
k2[:,1] = -1 + np.cos(0.2 * u - 8) * np.sqrt(u + 10)

w_hid_o = (np.random.rand(50) - 0.5) * 10
rbf = (np.random.rand(50, 2) -0.5) * 30
learningRate = 0.005



def f_act(x):
    return np.exp(-(x ** 2))

def net(x):
    global fact, w_hid_o, rbf
    fact = f_act(np.linalg.norm(x - rbf, axis = 1))
    return np.dot(w_hid_o, fact)

def train(t, x):
    global learningRate, fact, w_hid_o
    delta_w = learningRate * (t - net(x)) * fact
    w_hid_o += delta_w

for i in range(40):
    for r in np.random.random_integers(0, 399, 400):
        if(r < 200):
            train(1, k1[r])
        else:
            train(-1, k2[r - 200])



plt.scatter(rbf[0][0], rbf[0][1])
plt.scatter(k1[:,0], k1[:,1])
plt.scatter(k2[:,0], k2[:,1])
plt.scatter(rbf[:,0], rbf[:,1])
plt.show()
