# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>
# http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
print(__doc__)

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets, decomposition


# -- Functions -- #
def plot_embedding(dataset, embedding, subplot_number, manifold_learning_method_name):
    Y = embedding.fit_transform(dataset)
    ax = fig.add_subplot(subplot_number)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title(manifold_learning_method_name)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

# -- Main -- #





# Next line to silence pyflakes. This import is needed.
Axes3D

n_samples = 1000
dataset, color = datasets.samples_generator.make_swiss_roll(n_samples)  #  .make_s_curve(n_samples, random_state=0)
print(datasets.samples_generator.make_swiss_roll(n_samples))
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(16, 9))
plt.suptitle("Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14)


ax = fig.add_subplot(251, projection='3d')
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)  #  Set the elevation and azimuth of the axes.


# -- Embedding for LLE -- #
methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    emb = manifold.LocallyLinearEmbedding(
                    n_neighbors,
                    n_components,
                    eigen_solver='auto',
                    method=method)

    plot_embedding(dataset, emb, 252 + i, labels[i])


# -- Embedding for Isomap -- #
isomap = manifold.Isomap(
                n_neighbors,
                n_components)
plot_embedding(dataset, isomap, 257, 'Isomap')

# -- Embedding for MDS -- #
mds = manifold.MDS(
                n_components,
                max_iter=100,
                n_init=1)
plot_embedding(dataset, mds, 258, 'MDS')


# -- Embedding for PCA -- #
pca = decomposition.PCA(n_components)
plot_embedding(dataset, pca, 259, 'PCA')



# -- Embedding for Spectral Embedding -- #
# se = manifold.SpectralEmbedding(
#                 n_components=n_components,
#                 n_neighbors=n_neighbors)
# plot_embedding(dataset, se, 259)

# -- Embedding for T-SNE -- #
# tsne = manifold.TSNE(
#                     n_components=n_components,
#                     init='pca',
#                     random_state=0)
# plot_embedding(dataset, tsne, 259)

plt.show()











