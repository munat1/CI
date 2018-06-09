import numpy as np
import matplotlib.pyplot as plt

U = range(1, 201)

Klasse11 = []
Klasse12 = []

Klasse21 = []
Klasse22 = []

for u in range(1, 201):
    Klasse11.append(2 + np.sin(0.2 * u + 8) * np.sqrt(u + 10))
    Klasse12.append(-1 + np.cos(0.2 * u + 8) * np.sqrt(u + 10))
    
    Klasse21.append(2 + np.sin(0.2 * u - 8) * np.sqrt(u + 10))
    Klasse22.append(-1 + np.cos(0.2 * u - 8) * np.sqrt(u + 10))

K11 = mlines.Line2D([], [], color='red', marker='o', markersize=5, label='Klasse 1 x1')
K12 = mlines.Line2D([], [], color='magenta', marker='o', markersize=5, label='Klasse 1 x2')
K21 = mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Klasse 2 x1')
K22 = mlines.Line2D([], [], color='cyan', marker='o', markersize=5, label='Klasse 2 x2')

plt.legend(handles=[K11, K12, K21, K22])

plt.axis([-50, 250, -30, 30])

plt.plot(U, Klasse11, 'ro', U, Klasse12, 'mo', U, Klasse21, 'bo', Klasse22, 'co', markersize=1)

# plt.savefig("graph.png", dpi=1000)

plt.show()

    
