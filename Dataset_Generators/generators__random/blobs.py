from sklearn import manifold, datasets
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Usage: python "Dataset_Generators/generators__random/blobs.py" -n 1000 -d 3 -seed 12 -blobs 1 -output "Data/Random/blobs_n-1000_d-3_blobs-1_seed-12.train" -if_color_col 0

# -- Used for small experiment to check modes or omega_0 -- #
# python "Dataset_Generators/generators__random/blobs.py" -n 10000 -d 256 -seed 1 -blobs 3 -output "Data/Clustered_Exp/Clustered_n-10000_d-256_blobs-3_seed-1.train" -if_color_col 0


def plot_2D(dataset_1, color="teal"):
    fig = plt.figure(figsize=(5, 5))
    point_size = 3

    # ---- First subplot
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlim([min_x_dim, max_x_dim])
    # ax.set_ylim([min_y_dim, max_y_dim])
    # set_axes_range_2D(ax, max_val_PC, min_val_PC)
    ax.scatter(
        dataset_1[:, 0],
        dataset_1[:, 1],
        c=color,
        s=point_size)

    plt.show()


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", help="n_samples", type=int, required=1)
    parser.add_argument("-d", help="n_dimensions", type=int, required=1)
    parser.add_argument("-seed", help="seed/random state", type=int, nargs='?', default=None)
    parser.add_argument("-blobs", help="number of blobs", type=int, nargs='?', const=1, default=1)
    parser.add_argument("-blob_std", help="stddev of each blob", type=float, nargs='?', const=1, default=1)
    parser.add_argument("-output", help="blobs_dataset output file", required=1)
    parser.add_argument("-if_color_col", help="color column attached when saving the generated data, where 0 means no and 1 means yes", type=int, required=1)
    args = parser.parse_args()

    return args


def main_blobs():
    args = read_args()

    n_samples = args.n
    n_dimensions = args.d
    seed = args.seed
    blobs = args.blobs
    blobs_std = args.blob_std
    output_file = args.output
    if_color_col = args.if_color_col
    dataset_blobs, color = datasets.samples_generator.make_blobs(
                                                            n_samples=n_samples,
                                                            n_features=n_dimensions,
                                                            cluster_std=blobs_std,
                                                            centers=blobs,
                                                            center_box=(-100.0, 100.0),
                                                            shuffle=True,
                                                            random_state=seed)

    # plot_2D(dataset_blobs)
    print(dataset_blobs)
    print(color)
    color_col = np.array([[x] for x in color])
    dataset_blobs_with_color_col = np.concatenate((dataset_blobs, color_col), axis=1)

    if if_color_col == 1:
        np.savetxt(output_file, dataset_blobs_with_color_col, delimiter=' ')
    else:
        np.savetxt(output_file, dataset_blobs, delimiter=' ')

    # -- Test making data with multivariate distribution -- #
    # test = np.random.multivariate_normal([1,1], [[0.3, 0.2],[0.2, 0.2]], n_samples)
    # np.savetxt(output_file, test, delimiter=' ')
    plot_2D(dataset_blobs)

    # # -- Test making random data -- #
    # random_data = np.random.rand(n_samples, n_dimensions)
    # np.savetxt(output_file, random_data, delimiter=' ')
    # plot_2D(random_data)

if __name__ == '__main__':
    main_blobs()


# Dog and Cancer examples: if I get asked 'is this a dog?', then I always answer 'yes', which means I always get to recall almost all the dogs, but I never get to 'match' only the good ones, meaning precision is going to be low
# [[ 0.0104411 ]
#  [ 0.00394713]
#  [ 0.002876  ]
#  [ 0.002876  ]
#  [ 0.002876  ]]
# [[ 0.99374131]
#  [ 0.99930459]
#  [ 1.        ]
#  [ 1.        ]
#  [ 1.        ]]
# [[ 0.02066507]
#  [ 0.0078632 ]
#  [ 0.0057355 ]
#  [ 0.0057355 ]
#  [ 0.0057355 ]]
