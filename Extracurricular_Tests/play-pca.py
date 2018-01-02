import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

from sklearn.decomposition import PCA

from helpers import normalize_data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def set_axes_range_2D(ax, max_val_PC, min_val_PC):
    ax.set_xlim(max_val_PC[0], min_val_PC[0])
    ax.set_ylim(max_val_PC[1], min_val_PC[1])

def plot_3subfigs_2D(dataset_1, dataset_2, dataset_3, max_val_PC, min_val_PC):
    fig = plt.figure(figsize=(15, 5))
    point_size = 3
    colors = ['#5d009f', '#3d3d3d', '#efefef']  # some kind of nice purple

    # ---- First subplot
    ax = fig.add_subplot(1, 3, 1)
    set_axes_range_2D(ax, max_val_PC, min_val_PC)
    ax.scatter(
        dataset_1[:, 0],
        dataset_1[:, 1],
        c=colors[0],
        s=point_size)

    # ---- Second subplot
    ax = fig.add_subplot(1, 3, 2)
    set_axes_range_2D(ax, max_val_PC, min_val_PC)
    ax.scatter(
        dataset_2[:, 0],
        dataset_2[:, 1],
        c=colors[1],
        s=point_size)

    # ---- Third subplot
    ax = fig.add_subplot(1, 3, 3)
    set_axes_range_2D(ax, max_val_PC, min_val_PC)
    ax.scatter(
        dataset_3[:, 0],
        dataset_3[:, 1],
        c=colors[1],
        s=point_size)

    plt.show()


def set_axes_range_3D(ax):
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])



def plot_3subfigs_3D(dataset_1, dataset_2, dataset_3):
    fig = plt.figure(figsize=(15, 5))
    point_size = 3
    colors = ['#5d009f', '#3d3d3d', '#efefef']  # some kind of nice purple

    # ---- First subplot
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    set_axes_range_3D(ax)
    ax.scatter(
        dataset_1[:, 0],
        dataset_1[:, 1],
        dataset_1[:, 2],
        c=colors[0],
        s=point_size)  # , cmap=plt.cm.Spectral

    # ---- Second subplot
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    set_axes_range_3D(ax)
    ax.scatter(
        dataset_2[:, 0],
        dataset_2[:, 1],
        dataset_2[:, 2],
        c=colors[1],
        s=point_size)  # , cmap=plt.cm.Spectral

    # ---- Third subplot
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    set_axes_range_3D(ax)
    ax.scatter(
        dataset_3[:, 0],
        dataset_3[:, 1],
        dataset_3[:, 2],
        c=colors[1],
        s=point_size)  # , cmap=plt.cm.Spectral

    plt.show()


def pca_as_algo(data_norm, bits_to_encode_to, data_norm_d):
    n_pca = min(bits_to_encode_to, data_norm_d)
    cov_mat = np.cov(data_norm, rowvar=False)
    # log_array_to_file(log_file_train, cov_mat, "cov_mat")
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    # NB: signs of pc are slightly different than in matlab!!!! => same for the signs in data_norm_pcaed
    pc = np.fliplr(eig_vecs[:, -n_pca:])  # flip array in left-right direction
    data_norm_pcaed = data_norm.dot(pc)
    return pc, data_norm_pcaed

# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=3,
                    shrinkA=0, shrinkB=0, color="#3d3d3d")
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def plot_pca_to_show_variance_introduction(X):
    # - 2. PCA the data - #
    pca = PCA(n_components=2)
    X_pca_lib = pca.fit_transform(X.copy())
    X_pca_algo = pca_as_algo(X.copy(), 2, 2)
    print("X_pca_algo")
    print(X_pca_algo)
    print("X_pca_lib")
    print(X_pca_lib)

    print(pca.components_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    # print(X_pca)
    # -- Visualize the eigenvectors -- #
    plt.scatter(X[:, 0], X[:, 1], alpha=1, color="#55A868")

    for length, vector in zip(pca.explained_variance_, pca.components_):
        print(length, vector)
        v = vector * np.sqrt(length) * 2 * (-1)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal');
    plt.xlabel("Dimension d = 0")
    plt.ylabel("Dimension d = 1")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.show()


def main():
    # -- Experiment Info: description && motivation && data -- #
    print("# -- This has the purpose of checking whether data after PCAed looks uniformly distributed on the new axes/PC. -- #")

    # -- Read original data -- #
    # delimiter = ' '
    # data_filename_location = "../Data/Handmade/h3.train"
    # data_train_original = np.genfromtxt(data_filename_location, delimiter=delimiter, dtype=np.float)
    # data_train_original_norm = normalize_data(data_train_original)

    rng = np.random.RandomState(1)
    data_train_original = np.dot(rng.rand(2, 2), rng.randn(2, 1000)).T
    data_train_original_norm = normalize_data(data_train_original)

    # print(data_train_original_norm)

    # -- Plot 2D data before PCA-ing it -- #

    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)
    data_train_original_norm_2D = data_train_original_norm[:, 0:2].copy()
    # print("# -- data_train_original_norm_2D -- #")
    # print(data_train_original_norm_2D)


    # -- Try stuff from: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html -- #
    # - 1. Get the data - #
    X = data_train_original_norm_2D
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.axis('equal');
    # plt.show()

    plot_pca_to_show_variance_introduction(X)


    # -- PCA as dimensionality reduction -- #
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)

    print("X: ", X)
    print("X_pca: ", X_pca)
    X_new = pca.inverse_transform(X_pca)
    print("X_new: ", X_new)

    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, color="#55A868", s=30)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=1)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.9, color="#C44F53", s=25)
    plt.axis('equal');
    plt.xlabel("Dimension d = 0")
    plt.ylabel("Dimension d = 1")
    plt.show()


main()

