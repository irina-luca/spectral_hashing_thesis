# -- This script follows tuts from: http://www.math.ucla.edu/~mathluo/demo_graph/demo1.html -- #
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import numpy as np
import sklearn
from sklearn import datasets
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
import matplotlib.pyplot as plt

from sklearn import manifold as m


# -- 1. Create Two Moons dataset with different noise levels -- #
n_samples = 800
x = np.zeros([n_samples, 2, 3])
y = np.zeros([n_samples, 3])
n_mid = int(n_samples/2)
color = 'r'*n_mid + 'g'*n_mid
number_of_subfigures = 3
points_size = 2
plt.figure(1, figsize=(9, 6))
for i in range(number_of_subfigures):
    noise = 0.02+.03*(i+1)
    x[:, :, i], y[:, i] = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise, shuffle=False)
    plt.subplot(1, number_of_subfigures, i+1)
    plt.scatter(x[:, 0, i], x[:, 1, i], s=points_size, color=color)
    plt.axis('off')
    plt.title('TwoMoons, sigma = {sig}'.format(sig=noise))
plt.tight_layout()
plt.rcParams['figure.figsize'] = (9, 5)
# plt.show()

# -- 2. We plot the spectral embedding of the three Two Moons dataset with different kernels -- #
# rbf nearest neighbor graph
plt.figure(2, figsize=(9, 6))
plt.rcParams['figure.figsize'] = (12, 14)
for i in range(3):
    for j in range(4):
        raw_data = x[:, :, i]
        gamma_noise = 0.02+.03*(i+1)
        spectral_embedding = m.SpectralEmbedding(n_components=2, affinity='rbf', gamma=gamma_noise, random_state = None, eigen_solver='arpack', n_neighbors=15+13*(j), n_jobs=1)
        # perform_spectral_embedding
        V = spectral_embedding.fit_transform(raw_data)
        x_t = V[:, 0]
        y_t = V[:, 1]
        plt.subplot(3, 4, i*4+j+1)
        plt.scatter(x_t, y_t, s=points_size, color=color)
        plt.axis('off')
        plt.title('NN, K = {K}, Sig = {sigma}'.format(K = 15+13*(j), sigma=gamma_noise))
plt.show()

