import matplotlib.pyplot as plt
import numpy as np
import random as rnd

class RBF:
    def __init__(self):

    #Input Data
        self.input_space = np.linspace(1, 200, 200)

        self.class_1_x1 = 2 + ( np.sin((0.2 * self.input_space) + 8) * np.sqrt(self.input_space + 10))
        self.class_1_x2 = -1 + ( np.cos((0.2 * self.input_space) + 8) * np.sqrt(self.input_space + 10))
        self.class_1_y = 1

        self.class_2_x1 = 2 + ( np.sin((0.2 * self.input_space) - 8) * np.sqrt(self.input_space + 10))
        self.class_2_x2 = -1 + ( np.cos((0.2 * self.input_space) - 8) * np.sqrt(self.input_space + 10))
        self.class_2_y = -1

    #Konstatnten
        self.input_dim = 2
        self.hidden_dim = 50
        self.output_dim = 1
        self.training_steps = np.linspace(-16, 16, 321)
        self.learn_rate = 0.1        


        self.input_weights = 32 * np.random.random((self.hidden_dim, self.input_dim)) - 16
        self.hidden_weights = np.ones((self.hidden_dim, self.output_dim))

        self.hidden_activation = np.zeros((self.output_dim, self.hidden_dim))

        self.deltas = list()
        self.deltas.append(np.zeros_like(self.input_weights))

    def plotGraph(self):
        plt.ion()
        plt.figure()
        plt.plot(self.class_1_x1,self.class_1_x2, label ='Klasse 1')
        plt.plot(self.class_2_x1,self.class_2_x2, label ='Klasse 2')
        x = self.input_weights[:,0]
        y = self.input_weights[:,1]
        plt.scatter(x, y, label ='Neurons')
        plt.legend()
        plt.show()

    def act(self, x):
        sig = 1
        return np.exp( (-(x*x))/(sig*sig*2) )

    def get_input(self, x):
        if(x[1] == 0):
            return [self.class_1_x1[x[0]], self.class_1_x2[x[0]]]
        else:
            return [self.class_2_x1[x[0]], self.class_2_x2[x[0]]]

    def train(self):
        training_set = rnd.sample(range(0, 199), 25)
        for p in range(25):
            training_set[p] = [training_set[p], rnd.randint(0,1)]
            if (training_set[p][1] == 0):
                training_set[p][1] = -1
        deltas = list()

        for position in range(25):
            x = self.get_input(training_set[position])
            y = training_set[position][1]
            delta_i = list()
            for i in range(self.hidden_dim):
                delta_i.append(np.ones_like(self.input_weights))
                for j in range(self.input_dim):
                    self.hidden_activation[0][i] += np.power(x[j] - self.input_weights[i][j], 2)
                self.hidden_activation[0][i] = self.act(np.sqrt(self.hidden_activation[0][i]))
 

            output = self.hidden_activation.dot(self.hidden_weights)[0][0]
            for i in range(self.hidden_dim):
                dist = np.sqrt(np.power(x[0] - self.input_weights[i][0], 2) + np.power(x[1] - self.input_weights[i][1], 2))
                delta_i[i] = (self.learn_rate * (y - output) * dist)
                if ((i == self.hidden_dim-1) & (position == 0)):
                    print(delta_i)
                    print(output)
                    print(y)
                    print('------------------')
            deltas.append(delta_i)
            if(position == 24):
                delta = np.zeros_like(delta_i)
                for i in range(25):
                    delta += deltas[i]
                self.input_weights += delta
        test.plotGraph()
test = RBF()
test.train()
print("Done")
