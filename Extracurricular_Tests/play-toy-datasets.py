# -- This script is initially taken from http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html -- #
print(__doc__)

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

import sklearn
sklearn.__version__


def generate_datasets(n_samples):
    noisy_circles, noisy_circles_labels = datasets\
        .make_circles(
            n_samples=n_samples,
            factor=.5,
            noise=.05
        )
    return noisy_circles, noisy_circles_labels


def plot_2d(dataset, dataset_labels):
    colors = ['r', 'g']
    plt.scatter(dataset[:, 0], dataset[:, 1], color=colors[dataset_labels].tolist(), s=10)
    plt.show()

def main():
    n_samples = 100
    dataset, dataset_labels = generate_datasets(n_samples)
    print(dataset)
    print(dataset_labels)
    plot_2d(dataset, dataset_labels)

main()