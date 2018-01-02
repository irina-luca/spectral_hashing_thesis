import numpy as np
import math
import random as rand
import scipy as sp
from scipy.spatial import distance
from scipy.spatial import distance_matrix

from scipy.sparse import *
from scipy import *
from distances import *
from sklearn.cluster import spectral_clustering
from sklearn.manifold import spectral_embedding
from sklearn import manifold as m
from sklearn import metrics

from scipy.sparse import csgraph

import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh


from numpy import linalg
from sklearn.decomposition import PCA

from helpers import *


def get_eigen_matrices(data_norm):
    distance_mat = distance_matrix(data_norm, data_norm)
    affinity_mat = np.exp(
        - distance_mat ** 2 / (2. * (distance_mat.max() - distance_mat.min()) ** 2))
    laplacian_mat = csgraph.laplacian(affinity_mat, normed=False)
    return distance_mat, affinity_mat, laplacian_mat

def get_lapl_mat_from_rbf_kernel(rbf_kernel):
    return csgraph.laplacian(rbf_kernel, normed=False)

def main_playing_stage():
    data_filename = 'h1'
    folder_path = '../Data/Handmade/'

    # Import dataset
    delimiter = ' '
    data_train = np.genfromtxt(folder_path + data_filename + '.train', delimiter=delimiter, dtype=np.float)

    # Get datasets' sizes
    n_train = data_train.shape[0]
    d = data_train.shape[1]

    n_pc = 2

    # Normalize data to unit hypercube
    data_train_norm = normalize_data(data_train)
    print("normalized data: data_train_norm")
    print(data_train_norm)

    cov_mat = np.cov(data_train_norm, rowvar=False)
    eig_vals_eigh, eig_vecs_eigh = np.linalg.eigh(cov_mat)
    print("eig_vals_eigh from cov matrix")
    print(eig_vals_eigh)
    print("eig_vecs_eigh from cov matrix")
    print(eig_vecs_eigh)

    # -- Take the eig_vecs corresponding to the biggest eig_vals -- #
    eig_vecs_eigh = np.fliplr(eig_vecs_eigh)[:, -d:-d+n_pc]
    data_train_norm_transformed = data_train_norm.dot(eig_vecs_eigh)
    print("data_train_norm_transformed")
    print(data_train_norm_transformed)

    # print("cov matrix for transformed data")
    # cov_mat_transformed = np.cov(data_train_norm_transformed, rowvar=False)
    # print(cov_mat_transformed)

    dist_mat, aff_mat, lapl_mat = get_eigen_matrices(data_train_norm)
    dist_mat_tr, aff_mat_tr, lapl_mat_tr = get_eigen_matrices(data_train_norm_transformed)
    print("eigen matrices for initial data: dist_mat, aff_mat, lapl_mat")
    # print(dist_mat)
    # print("aff_mat")
    # print(aff_mat)
    # print("lapl_mat")
    # print(lapl_mat)
    print("eigen matrices for transformed data: dist_mat, aff_mat, lapl_mat")
    # print(dist_mat_tr)
    # print("aff_mat_tr")
    # print(aff_mat_tr)
    # print("lapl_mat_tr")
    # print(lapl_mat_tr)
    #
    # print("test with rbf gaussian kernel in python")
    rbf_kernel = metrics.pairwise.rbf_kernel(data_train_norm, data_train_norm)
    rbf_kernel_tr = metrics.pairwise.rbf_kernel(data_train_norm_transformed, data_train_norm_transformed)

    # print("rbf_kernel")
    # print(rbf_kernel)
    # print("rbf_kernel_tr")
    # print(rbf_kernel_tr)
    # print("laplacian from rbf_kernel with data_train_norm")
    lapl_rbf = get_lapl_mat_from_rbf_kernel(rbf_kernel)
    # print(lapl_rbf)
    # print("laplacian from rbf_kernel with data_train_norm_transformed")
    lapl_rbf_tr = get_lapl_mat_from_rbf_kernel(rbf_kernel_tr)
    # print(lapl_rbf_tr)


    spectral_emb = m.SpectralEmbedding(n_components=2, affinity='rbf', gamma = None, random_state = None, eigen_solver = None, n_neighbors = 3, n_jobs = 1)
    print(spectral_emb.fit_transform(data_train_norm))
    print(spectral_emb.fit_transform(data_train_norm_transformed))





    # U, s, V = np.linalg.svd(lapl_rbf_tr, full_matrices=True)
    # print(U)
    # print(s)
    # print(V)




    # print(spectral_emb.affinity_matrix_)
    # eig_vals_eigh_tr, eig_vecs_eigh_tr = np.linalg.eigh(cov_mat_transformed)
    # print("eig_vals for transformed training set")
    # print(eig_vals_eigh_tr)
    # print("eig_vecs for transformed training set")
    # print(eig_vecs_eigh_tr)















    # print(np.cov(data_train_norm_transformed[:, 1:3], rowvar=False))
    # distance_matrix_train = distance_matrix(data_train_norm, data_tr2ain_norm)
    # affinity_matrix_train = np.exp(- distance_matrix_train ** 2 / (2. * (distance_matrix_train.max()-distance_matrix_train.min()) ** 2))
    # print("affinity_matrix_train")
    # print(affinity_matrix_train)
    # laplacian_matrix_train = csgraph.laplacian(affinity_matrix_train, normed=False)
    #
    # print("-----------------------------------------------------------------------------")
    # print("laplacian_matrix_train")
    # print(laplacian_matrix_train)
    #
    #
    # print("-----------------------------------------------------------------------------")
    # print("Explore stuff from Laplacian")
    # eig_vals_L, eig_vect_L = np.linalg.eig(laplacian_matrix_train)
    # print("eig_vals_L")
    # print(eig_vals_L)
    # print("eig_vect_L")
    # print(eig_vect_L)
    #
    # print("")
    #
    # w, v = eigh(laplacian_matrix_train)  # w = eigenvalues, v = eigenvectors
    # print(w)
    # print(v)
    #
    #
    # print(" ")
    #
    # pca = PCA(n_components=3)
    # pca.fit_transform(data_train_norm)
    # print(pca.components_)
    #
    #
    #
    # print(" ")
    #
    # U, s, V = np.linalg.svd(laplacian_matrix_train, full_matrices=True)
    # print("U", U)
    # print("s", s)
    # print("V", V)


    # embedding = m.spectral_embedding(laplacian_matrix_train, n_components=2)
    # print(embedding)
    # print(cov(data_train_norm, rowvar=False))
    # print(pca.singular_values_)
    # print(pca.singular_values_)
    # print("Explore PC of the dataset")
    # # transformed_data_train = data_train_norm.dot(eig_vect_L)
    # # print("transformed_data_train", transformed_data_train)
    #
    # A = affinity_matrix_train
    # D = np.diag(np.sum(A, 0))
    # D_half_inv = np.diag(1.0 / np.sqrt(np.sum(A, 0)))
    # M = np.dot(D_half_inv, np.dot((D - A), D_half_inv))
    # # compute eigenvectors and eigenvalues
    # (w, v) = np.linalg.eigh(M)
    # print(w)
    # print(v)
    #
    # print("-----------------------------------------------------------------------------")
    # print("Explore stuff from Similarity Matrix")
    # eig_vals_W, eig_vect_W = np.linalg.eig(affinity_matrix_train)
    # print("eig_vals_W")
    # print(eig_vals_W)
    # print("eig_vect_W")
    # print(eig_vect_W)



    # # eig_vals, eig_vecs = sp.sparse.linalg.eigs(affinity_matrix_train, k=4)
    # eig_vals_L, eig_vecs_L = linalg.eig(laplacian_matrix_train)
    # eig_vals_W, eig_vecs_W = linalg.eig(affinity_matrix_train)
    # print(eig_vals_L)
    # print(eig_vals_W)
    # # print(eig_vecs)
    #
    # spectral_embedding = m.spectral_embedding(laplacian_matrix_train, n_components=4, norm_laplacian=False)
    # print("spectral_embedding", spectral_embedding)
    #
    # # print(cov(data_train_norm, rowvar=False))


main_playing_stage()

