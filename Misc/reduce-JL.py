import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances





def get_JL_n_components_by_bounds(n_samples):  #  no clue if this is right at all
    eps = 0.474 # np.finfo(float).eps
    n_components = 4 * np.log(n_samples) / (np.power(eps, 2) / 2 - np.power(eps, 3) / 3)
    return format(n_components, 'f')



# Part 0: get n_components by bounds declared in the JL lemma, for our 20newsgroups dataset ((18846, 101631)) =>
# n_JL_components = get_JL_n_components_by_bounds(18846)
# print(n_JL_components)

# # Part 1: plot the theoretical dependency between n_components_min and
# # n_samples
#
# # range of admissible distortions
# eps_range = np.linspace(0.1, 0.99, 5)
# colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))
#
# # range of number of samples (observation) to embed
# n_samples_range = np.logspace(1, 9, 9)
#
# plt.figure()
# for eps, color in zip(eps_range, colors):
#     min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
#     plt.loglog(n_samples_range, min_n_components, color=color)
#
# plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
# plt.xlabel("Number of observations to eps-embed")
# plt.ylabel("Minimum number of dimensions")
# plt.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components")
# # plt.show()
#
# # range of admissible distortions
# eps_range = np.linspace(0.01, 0.99, 100)
#
# # range of number of samples (observation) to embed
# n_samples_range = np.logspace(2, 6, 5)
# colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))
#
# plt.figure()
# for n_samples, color in zip(n_samples_range, colors):
#     min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
#     plt.semilogy(eps_range, min_n_components, color=color)
#
# plt.legend(["n_samples = %d" % n for n in n_samples_range], loc="upper right")
# plt.xlabel("Distortion eps")
# plt.ylabel("Minimum number of dimensions")
# plt.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")

# Part 2: perform sparse random projection of some digits images which are
# quite low dimensional and dense or documents of the 20 newsgroups dataset
# which is both high dimensional and sparse

# if '--twenty-newsgroups' in sys.argv:
#     # Need an internet connection hence not enabled by default
#     data = fetch_20newsgroups_vectorized().data[:500]
# else:
#     data = load_digits().data[:500]







data = fetch_20newsgroups_vectorized(subset='all', remove=('headers', 'footers', 'quotes')).data.todense()

n_samples, n_features = data.shape
print("Embedding %d samples with dim %d using various random projections"
      % (n_samples, n_features))

# n_components_range = np.array([300, 1000, 10000])
# n_components_range = np.array([300, 1000, 10000])
# dists = euclidean_distances(data, squared=True).ravel()

# select only non-identical samples pairs
# nonzero = dists != 0
# dists = dists[nonzero]

# N.B.: !!! if eps=0.474, n_comp=512. if eps='auto'=0.1, then n_comp~8000 dimensions. Try all the variants


# for n_components in n_components_range:
t0 = time()
rp = SparseRandomProjection(n_components='auto')  # n_components
print(rp.n_components)
print(rp.eps)
projected_data = rp.fit_transform(data)
projected_data_test = rp.transform(data)
print(projected_data_test)
print(projected_data_test.shape)
# print(projected_data_test.data.shape)
# print(projected_data.data)
# print(projected_data.data.todense())
# print(projected_data.data)
# print(rp.components_)
# n_JL_components = projected_data.shape[1]
# print("Projected %d samples from %d to %d in %0.3fs"
#       % (n_samples, n_features, n_JL_components, time() - t0))
# # if hasattr(rp, 'components_'):
# #     n_bytes = rp.components_.data.nbytes
# #     n_bytes += rp.components_.indices.nbytes
# #     print("Random matrix with size: %0.3fMB" % (n_bytes / 1e6))
#
# projected_dists = euclidean_distances(
#     projected_data, squared=True).ravel()[nonzero]
#
# plt.figure()
# plt.hexbin(dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu)
# plt.xlabel("Pairwise squared distances in original space")
# plt.ylabel("Pairwise squared distances in projected space")
# plt.title("Pairwise distances distribution for n_components=%d" %
#           n_JL_components)
# cb = plt.colorbar()
# cb.set_label('Sample pairs counts')
#
# rates = projected_dists / dists
# print("Mean distances rate: %0.2f (%0.2f)"
#       % (np.mean(rates), np.std(rates)))
#
# plt.figure()
# plt.hist(rates, bins=50, normed=True, range=(0., 2.), edgecolor='k')
# plt.xlabel("Squared distances rate: projected / original")
# plt.ylabel("Distribution of samples pairs")
# plt.title("Histogram of pairwise distance rates for n_components=%d" %
#           n_JL_components)
#
# # TODO: compute the expected value of eps and add them to the previous plot
# # as vertical lines / region
#
# plt.show()