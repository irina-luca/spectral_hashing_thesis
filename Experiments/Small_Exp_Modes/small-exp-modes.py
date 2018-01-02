import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

from helpers import normalize_data


def set_axes_range_2D(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def plot_3subfigs_2D(dataset_1, dataset_2, dataset_3):
    fig = plt.figure(figsize=(15, 5))
    point_size = 3
    colors = ['#5d009f', '#3d3d3d', '#efefef']  # some kind of nice purple

    # ---- First subplot
    ax = fig.add_subplot(1, 3, 1)
    set_axes_range_2D(ax)
    ax.scatter(
        dataset_1[:, 0],
        dataset_1[:, 1],
        c=colors[0],
        s=point_size)

    # ---- Second subplot
    ax = fig.add_subplot(1, 3, 2)
    set_axes_range_2D(ax)
    ax.scatter(
        dataset_2[:, 0],
        dataset_2[:, 1],
        c=colors[1],
        s=point_size)

    # ---- Third subplot
    ax = fig.add_subplot(1, 3, 3)
    set_axes_range_2D(ax)
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


def save_variants_to_files(output_destination, dataset_variant_1, dataset_variant_2, dataset_variant_3):
    np.savetxt(output_destination + ".train", dataset_variant_1, delimiter=' ')
    np.savetxt(output_destination + "_squeezed_1.train", dataset_variant_2, delimiter=' ')
    np.savetxt(output_destination + "_squeezed_2.train", dataset_variant_3, delimiter=' ')



def main():
    # -- Experiment Info: description && motivation && data -- #
    print("# -- This is a small experiment on a 3D -> 2D dataset, such that one dimension gets gradually squeezed. -- #")
    print("# -- The plan is to see how modes get affected, as well as how omega_zero reacts -- #")
    print("# -- The starting point targets dataset from file => \"Data/Random/Small_Exp_Modes/blobs_n-100_d-3_blobs-1_seed-1.train\". -- #")


    # -- Read original data -- #
    delimiter = ' '
    data_filename_location = "../../Data/Random/Small_Exp_Modes/blobs_n-100_d-3_blobs-1_seed-1.train"
    data_train_original = np.genfromtxt(data_filename_location, delimiter=delimiter, dtype=np.float)
    data_train_original_norm = normalize_data(data_train_original)

    # data_train_original_norm__squeezed_1 = data_train_original_norm.copy()
    # data_train_original_norm__squeezed_1[:, 0] *= 390.0
    # data_train_original_norm__squeezed_1[:, 1] = 1
    # data_train_original_norm__squeezed_1[:, 2] = 0.5
    #
    # data_train_original_norm__squeezed_2 = data_train_original_norm.copy()
    # data_train_original_norm__squeezed_2[:, 0] *= 0.9
    # data_train_original_norm__squeezed_2[:, 1] *= 0.1
    # data_train_original_norm__squeezed_2[:, 2] *= 0.5

    # -- Make a simpler example, from 2D -> 2D -- #
    data_train_original_norm__v1 = data_train_original_norm[:, 0:2].copy()
    data_train_original_norm__v2 = data_train_original_norm[:, 0:2].copy()

    # data_train_original_norm__v2[:, 0] *= 0.1
    data_train_original_norm__v2[:, 0] *= 0.00000001

    plot_3subfigs_2D(data_train_original_norm__v1, data_train_original_norm__v2, data_train_original_norm__v1)

    # -- Generate gradually squeezed data and plot all variants in 3D -- #
    # plot_3subfigs_3D(data_train_original_norm, data_train_original_norm__squeezed_1, data_train_original_norm__squeezed_2)
    # plot_3subfigs_2D(data_train_original_norm, data_train_original_norm__squeezed_1, data_train_original_norm__squeezed_2)

    # print(data_train_original_norm[0])
    # print(data_train_original_norm__squeezed_1[0])
    # print(data_train_original_norm__squeezed_2[0])

    # -- Save the all the variants (3, so far) to files so I can train them afterwards -- #
    # output_destination = "../../Data/Random/Small_Exp_Modes/blobs_n-100_d-3_blobs-1_seed-1"
    # save_variants_to_files(output_destination, data_train_original_norm, data_train_original_norm__squeezed_1, data_train_original_norm__squeezed_2)

    output_destination = "../../Data/Random/Small_Exp_Modes/blobs_n-100_d-2_blobs-1_seed-1"
    np.savetxt(output_destination + "_unsqueezed.train", data_train_original_norm__v1, delimiter=' ')
    np.savetxt(output_destination + "_squeezed.train", data_train_original_norm__v2, delimiter=' ')

main()

