import itertools
import matplotlib.pyplot as plt
import numpy as np
from sympy.combinatorics.graycode import GrayCode
from scipy import interpolate
from distances import dist_hamming
import plotly.plotly as py
import matplotlib.font_manager as font_manager
import plotly.graph_objs as go
import seaborn as sns

def normalize_array(array):
    norm_array = [(value - min(array)) / (max(array) - min(array) + 0.000001) for value in array]
    return norm_array

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_sin(mode_value, curve_splits, color, do_plot, cut_th):
    plt.xlim([0, 1])
    plt.ylim([-1, 1])
    x = normalize_array(np.arange(curve_splits))
    x_range = max(x) - min(x)
    # Calculate val/amplitude of the sin for each given sample in x
    y = [np.sin(np.pi / 2.0 + mode_value * np.pi * i / x_range) for i in x]

    opaque_values = [0, 2, 4]
    alpha_value = 1 if cut_th in opaque_values else 0.3
    # plt.stem(x, y, 'wheat')
    if do_plot:
        plt.plot(x, y, color=color, linewidth=1.5, alpha=alpha_value)

    return y

def plot_x_axis():
    plt.axhline(y=0.0, color='#efefef', linestyle=':-', linewidth=0.5)

def plot_data_x(num_data_coords):
    x_data_func = np.linspace(0, 1, int(num_data_coords))
    # print("x_data_func")
    # print(x_data_func)
    y_func_null = np.zeros(len(x_data_func))
    # print(y_func_null)
    # print("y_func_null")
    # print(y_func_null)

    plt.plot(x_data_func, y_func_null, '-', color='#efefef')

    return x_data_func


def plot_individual_sine_cuts(x_data_function, intersection_indices, sine_color):
    for ith, index in enumerate(intersection_indices):
        plt.axvline(x=x_data_function[index],
                    color=sine_color,
                    linewidth=5,
                    linestyle="--")


def plot_bar_chart(data, common_label, bar_width):
    num_items = len(data)
    bar_chart = plt.figure(figsize=(19, 3))

    ind = np.arange(num_items)
    plt.bar(ind, data, width=bar_width)
    plt.xticks(ind + bar_width / 2, [common_label + str(item) for item in range(1, num_items + 1)])
    bar_chart.autofmt_xdate()
    plt.ylim([0, 1])


def partition_box_into_buckets(show_x_axis, num_of_cuts, cuts_modes, show_sines, show_buckets_hashcodes, plot_buckets_sizes):
    # -- Establish/ Define some params -- #
    max_mode = max(cuts_modes)
    # sines_color_map = get_cmap(num_of_cuts)  # num_of_cuts is basically the # of sine functions we pass
    sines_color_map = ['#C44F53','#4D73B1', '#55A868', '#CEBD7E', '#5E80B8', '#9D92C3', '#3d3d3d']
    print("sines_color_map", sines_color_map)
    all_intersection_coords_indices = []
    intersection_coords_indices_for_each_cut = {}

    # -- Plot x-axis if condition -- #
    if show_x_axis:
        plot_x_axis()

    # -- Plot 'random' data on x-axis -- #
    num_samples = max_mode * 100
    x_data_function = plot_data_x(num_samples)

    # Sample rate (how many samples 'mark' each frequency curve)
    curve_splits = max_mode * 100
    for cut_th in range(0, num_of_cuts):
        print("cut_th")
        print(cut_th)
        # -- Plot individual sine SH function -- #
        mode_value = cuts_modes[cut_th]
        sine_color = sines_color_map[cut_th]
        print("#### sine color ###", sine_color)
        # if plot_sines:
        sine_function_for_cut_th = plot_sin(mode_value, curve_splits, sine_color, show_sines, cut_th)

        # -- Find intersection points between sin_function and x-axis (y=0 function, in fact) -- #
        intersection_indices_for_cut_th = np.argwhere(np.diff(np.sign(sine_function_for_cut_th)) != 0).reshape(-1) + 0
        all_intersection_coords_indices.extend(intersection_indices_for_cut_th)
        # Obs.: num_sine_partitions_for_cut_th must be equal to mode_value
        # num_sine_partitions_for_cut_th = len(intersection_indices_for_cut_th)
        intersection_coords_indices_for_each_cut[cut_th] = sorted(np.hstack(([0, num_samples], intersection_indices_for_cut_th)))
        # -- Plot all vertical functions x=const., corresponding to all the line cuts for the individual sine function found above (the intersections) -- #
        plot_individual_sine_cuts(x_data_function, intersection_indices_for_cut_th, sine_color)

    # -- Sort intersection coords -- #
    all_intersection_coords_indices.extend([0, num_samples])  # Extend with min and max values of the box as well
    all_intersection_coords_indices.sort()

    # -- After getting all the intersection points on x-axis, find out buckets' sizes -- #
    buckets_sizes_absolute = np.array([x - all_intersection_coords_indices[i - 1] for i, x in enumerate(all_intersection_coords_indices)][1:])
    buckets_sizes_relative = np.divide(buckets_sizes_absolute, (num_samples * 1.0))

    non_empty_buckets_sizes_relative = buckets_sizes_relative[np.nonzero(buckets_sizes_relative)]
    non_empty_buckets_sizes_absolute = buckets_sizes_absolute[np.nonzero(buckets_sizes_absolute)]
    max_bucket_size = max(non_empty_buckets_sizes_relative)
    min_bucket_size = min(non_empty_buckets_sizes_relative)

    print("max_bucket_size => ", max_bucket_size)
    print("min_bucket_size => ", min_bucket_size)
    print("num of non empty buckets => ", len(non_empty_buckets_sizes_relative))
    print("non_empty_buckets_sizes_relative => ", non_empty_buckets_sizes_relative)
    print("non_empty_buckets_sizes_relative ./ min_bucket_size (how many times each bucket is bigger than the smallest bucket) => ", [int(bucket / min_bucket_size) for bucket in non_empty_buckets_sizes_relative])
    print("order of magnitude between max bucket size vs. min bucket size => ", int(max_bucket_size / min_bucket_size))


    if plot_buckets_sizes:
        x = np.around(non_empty_buckets_sizes_relative, decimals=2)
        plot_bar_chart(x, "B_", 0.7)

    # -- Generate hash codes for the formed buckets -- #
    if show_buckets_hashcodes:
        unique_and_mid_intersection_coords_indices = np.unique(all_intersection_coords_indices)[1:-1]

        buckets_hashcodes = []
        for umi_th, unique_mid_intersection in enumerate(np.hstack((unique_and_mid_intersection_coords_indices, [num_samples]))):
            # print("------------------")
            # print("unique_mid_intersection is => ")
            # print(unique_mid_intersection)
            bucket_hashcode = []
            for pc_th, pc_cut_intersections in intersection_coords_indices_for_each_cut.items():
                intersections_of_provenance_for_unique_mid_intersection = [val < unique_mid_intersection for val in pc_cut_intersections]
                partition_tuples = [(i, int_of_prov != intersections_of_provenance_for_unique_mid_intersection[i - 1]) for i, int_of_prov in enumerate(intersections_of_provenance_for_unique_mid_intersection)][1:]
                provenance_partition = [item[0] for item in partition_tuples if item[1]]
                if provenance_partition[0] % 2 != 0:
                    bucket_hashcode.append(1)
                else:
                    bucket_hashcode.append(0)
            # print("------------------")
            buckets_hashcodes.append(bucket_hashcode)
        buckets_hashcodes_int = np.array(buckets_hashcodes, dtype=int)
        hashcodes = [''.join(str(bit) for bit in binary_hash) for binary_hash in buckets_hashcodes_int]
        print("generated buckets' hashcodes are => ")
        print(hashcodes)
        print("number of buckets is => ")
        print(len(hashcodes))

        # -- Check Hamming distance between all neighbouring buckets -- #
        dist_hamming_between_neighboring_buckets = [dist_hamming(buckets_hashcodes[b_hc - 1], bucket_hashcode) for b_hc, bucket_hashcode in enumerate(buckets_hashcodes)][1:]
        print("Hamming distances between all neighboring buckets are => ")
        print(dist_hamming_between_neighboring_buckets)

    return buckets_sizes_absolute, buckets_sizes_relative

def main():
    sns.set(style="darkgrid", color_codes=True, font_scale=2)
    # plt.figure(figsize=(20, 3))
    plt.xlim([0, 1])
    plt.ylim([-1, 1])
    plt.xlabel("PCs' scores")
    plt.ylabel("Eigenfunction values")

    # -- Partition box into buckets, depending on configuration params -- #
    max_mode = 4
    cuts_modes = [v for v in range(1, max_mode + 1)]
    num_of_cuts = len(cuts_modes)
    show_x_axis = False
    show_sines = True
    show_buckets_hashcodes = True
    plot_buckets_sizes = True
    partition_box_into_buckets(show_x_axis, num_of_cuts, cuts_modes, show_sines, show_buckets_hashcodes, plot_buckets_sizes)


    plt.show()





main()
