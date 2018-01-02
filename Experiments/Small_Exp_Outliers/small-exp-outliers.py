from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

from helpers import normalize_data

from sklearn import manifold, datasets
import argparse
import numpy as np



def plot_2D(dataset_1):
    fig = plt.figure(figsize=(5, 5))
    point_size = 3
    colors = ['#5d009f', '#3d3d3d', '#efefef']  # some kind of nice purple

    # ---- First subplot
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlim([min_x_dim, max_x_dim])
    # ax.set_ylim([min_y_dim, max_y_dim])
    # set_axes_range_2D(ax, max_val_PC, min_val_PC)
    ax.scatter(
        dataset_1[:, 0],
        dataset_1[:, 1],
        c=colors[0],
        s=point_size)

    plt.show()


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def visualize_PCAed_data_more_versions(dataset_blobs_norm):
    plot_2D(dataset_blobs_norm)
    # -- PCA from 2D -> 1D -- #
    pca = PCA(n_components=1)
    dataset_blobs_norm_PCAed = pca.fit_transform(dataset_blobs_norm)

    zeros_col = np.zeros((len(dataset_blobs_norm_PCAed), 1))
    dataset_blobs_norm_PCAed_with_zeros_col = np.hstack(
        [dataset_blobs_norm_PCAed, np.zeros((len(dataset_blobs_norm_PCAed), 1))])
    print(np.hstack([dataset_blobs_norm_PCAed, np.zeros((len(dataset_blobs_norm_PCAed), 1))]))
    plot_2D(normalize_data(dataset_blobs_norm_PCAed_with_zeros_col))

    # -- PCA from 2D -- #
    pca = PCA(n_components=2)
    dataset_blobs_norm_PCAed_2D = pca.fit_transform(dataset_blobs_norm)
    plot_2D(normalize_data(dataset_blobs_norm_PCAed_2D))


def visualize_PC_1D_axis(dataset_blobs_norm):
    X = dataset_blobs_norm.copy()
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)

    X_new = pca.inverse_transform(X_pca)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal');
    plt.show()

def rotate_data(data, angle):
    theta = (angle / 180.) * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    return np.dot(data, rot_matrix)

def visualize_PC_1D_axis_with_rotated_blob(dataset_blobs_for_rotation, outlier_1):
    # for angle = 243., the created blob with 1000 samples is perfectly round (or 244. and 10 000 samples)
    # for angle = 13., the created blob with 10 000 samples projects the outlier on the PC axis


    dataset_blobs_rotated = rotate_data(dataset_blobs_for_rotation.copy(), 243.)
    dataset_blobs_outlier_rotated = np.vstack([dataset_blobs_rotated.copy(), outlier_1])
    dataset_blobs_outlier_rotated_norm = normalize_data(dataset_blobs_outlier_rotated)
    print(dataset_blobs_outlier_rotated_norm.shape)

    plot_2D(dataset_blobs_outlier_rotated_norm)
    visualize_PC_1D_axis(dataset_blobs_outlier_rotated_norm.copy())

    return dataset_blobs_outlier_rotated


def make_testing_file_for_profi_vs_blob_outlier(n_samples, n_dimensions, blobs):
    testing_data, color_testing = datasets.samples_generator.make_blobs(
        n_samples=int(n_samples / 10),
        n_features=n_dimensions,
        cluster_std=1,
        centers=blobs,
        shuffle=True,
        random_state=13)

    testing_data_rotated = rotate_data(testing_data.copy(), 100.)

    print(testing_data_rotated.shape)
    plot_2D(normalize_data(testing_data_rotated))

    output_file = "../../Results/Experiments/Small_Exp_Outliers/blob-outlier_testing-on-1000-perfectly-round_ss=100.test"
    np.savetxt(output_file, testing_data_rotated, delimiter=' ')


def make_blob_outlier_data(n_samples, n_dimensions, blobs_std, blobs, seed, outlier):
    dataset_blobs, color = datasets.samples_generator.make_blobs(
        n_samples=n_samples,
        n_features=n_dimensions,
        cluster_std=blobs_std,
        centers=blobs,
        shuffle=True,
        random_state=seed)

    dataset_blobs_united = np.vstack([dataset_blobs.copy(), outlier])
    return dataset_blobs_united


def make_blob_data(n_samples, n_dimensions, blobs_std, blobs, seed):
    dataset_blobs, color = datasets.samples_generator.make_blobs(
        n_samples=n_samples,
        n_features=n_dimensions,
        cluster_std=blobs_std,
        centers=blobs,
        shuffle=True,
        random_state=seed)

    return dataset_blobs

def save_data_to_file(data, filename):
    output_file = "../../Results/Experiments/Small_Exp_Outliers/Data/" + filename
    np.savetxt(output_file, data, delimiter=' ')

def main():
    # -- Experiment Info: description && motivation && data -- #
    print("# -- This is a small experiment 2D generated/artificial dataset which should contain a blob and an outlier. -- #")
    print("# -- The plan is to see how PCAed data looks and also test SH on it by encoding to 2 bits. -- #")

    # -- Step 1: Create 2D blob-- #
    n_samples = 1000
    n_dimensions = 2
    blobs_std = 0.5
    blobs = 1
    seed = 1
    dataset_blobs, color = datasets.samples_generator.make_blobs(
        n_samples=n_samples,
        n_features=n_dimensions,
        cluster_std=blobs_std,
        centers=blobs,
        shuffle=True,
        random_state=seed)

    print("dataset_blobs => ")
    print(dataset_blobs)
    dataset_blobs_for_rotation = dataset_blobs.copy()
    # -- Step 2: Add outlier to the data -- #
    outlier_1 = [10, 10]
    dataset_blobs = np.vstack([dataset_blobs.copy(), outlier_1])
    # print("dataset_blobs with outlier => ")
    # print(dataset_blobs)

    dataset_blobs_norm = normalize_data(dataset_blobs)
    output_file = "../../Results/Experiments/Small_Exp_Outliers/blob_outlier_1.data"
    # np.savetxt(output_file, dataset_blobs, delimiter=' ')



    # -- *** See what PC axes PCA picks in 2 cases: 2D -> 2D and 2D -> 1D *** -- #
    # visualize_PCAed_data_more_versions(dataset_blobs_norm.copy())



    # -- *** See what PC axes PCA picks in 2 cases: 2D -> 2D and 2D -> 1D, in case I rotate that blob *** -- #
    # (Point): If I increase the number of points in the blob, the PC axis will rotate as well, until at some point, that outlier won't even matter anymore, cause it will get projected itself on the chosen PC axis
    # visualize_PC_1D_axis(dataset_blobs_norm.copy())




    # -- *** See in case 2D -> 1D how PC axis is chosen *** -- #
    # (Point): The more round the blob is, the farther apart the outlier and the blob will be.
    dataset_blobs_outlier_rotated = visualize_PC_1D_axis_with_rotated_blob(dataset_blobs_for_rotation.copy(), outlier_1)

    # output_file = "../../Results/Experiments/Small_Exp_Outliers/blob-outlier_perfectly-round_ss=1000.train"
    # np.savetxt(output_file, dataset_blobs_outlier_rotated, delimiter=' ')



    # -- Try to test ss=1000, k=100, #bits=2, testing_size=100 and compare with Profi, for both eval_types=0,1 -- #
    # make_testing_file_for_profi_vs_blob_outlier(n_samples, n_dimensions, blobs)


    # -- Make more datasets for the same purpose, but same dimensionality as Profi -- #
    # profi_dim = 4096
    # outlier = np.full((1, profi_dim), 10000)[0]
    # data_train = make_blob_outlier_data(1000, profi_dim, 0.5, 1, 1, outlier) # args: (n_samples, n_dimensions, blobs_std, blobs, seed)
    # data_test = make_blob_data(100, profi_dim, 0.9, 1, 1)
    #
    # print(data_test.shape)
    # save_data_to_file(data_train, "blob-outlier_ss=1000_d=4096.train")
    # save_data_to_file(data_test, "blob-outlier_ss=100_d=4096.test")

main()

