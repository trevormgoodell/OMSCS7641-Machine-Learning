import util
import matplotlib.pyplot as plt
import numpy as np

from itertools import permutations
import pandas as pd


X = np.arange(4)

# Wine KM Credit KM Wine EM Credit Em
original = [0.73, 0.78, 0.73, 0.78]
pca = [0.75, 0.81, 0.75, 0.81]
ica = [0.74, 0.81, 0.74, 0.81]
rp = [0.73, 0.8, 0.74, 0.79]
nnmf = [0.74, 0.8, 0.74, 0.8]

space = 1.0/6
plt.bar(X + -1*space, original, color='blue', width = space, label="Original")
plt.bar(X + 0*space, pca, color='green', width = space, label="PCA")
plt.bar(X + 1*space, ica, color='red', width = space, label="ICA")
plt.bar(X + 2*space, rp, color='purple', width = space, label="RP")
plt.bar(X + 3*space, nnmf, color='black', width = space, label="NNMF")

plt.legend(loc=4)

plt.xticks([r + space for r in range(4)],
        ['Accuracy KM', 'F1 Scores KM', 'Accuracy EM', 'F1 Scores EM'])


plt.show()
