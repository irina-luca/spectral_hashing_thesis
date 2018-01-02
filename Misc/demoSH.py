import datetime

import numpy as np
import math
import json
import random as rand
import argparse
from scipy.spatial import distance
from scipy.sparse import *
from scipy import *
from sklearn.neighbors import NearestNeighbors

from shparams import SHParam
from shparams import SHModel
from distances import *
from helpers import *
from evaluate import *

import time

# -- OBSERVATIONS -- #
# (1) Argument -i is data_filename for both testing and training, without extensions '.train', respectively '.test'. This occurs only for demoSH.py, as
#     it contains training and testing combined.


# -- Arguments as input to the script -- #
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file name (same for both testing and training so far)", required=1)
    parser.add_argument("-k", help="average number of nearest neighbours", type=int, required=1)
    parser.add_argument("-mhd", help="maximum Hamming distance used for testing(in evaluation)", type=int, required=1)
    parser.add_argument("-nost", help="number of splits/batches for testing set when calculating the ground truth",
                        type=int, nargs='?', default=10)
    parser.add_argument('-loopbits', '--loopbits', help='list input of bit code sizes to hash to/loopbits', type=str)
    parser.add_argument('-log_train', help='log file containing needed info after training', required=1, type=str)
    parser.add_argument("-log_test", help="log file for testing", type=str, required=1)

    args = parser.parse_args()

    return args



def train_sh(data_norm, bits_to_encode_to, data_train_filename_location, log_file_train_destination):
    start = time.time()
    # -- Setup log file to save important info after training -- #
    with open(log_file_train_destination, 'w') as log_file_train:
        # -- Get training set dimensions -- #
        log_file_train.write("# -- Get training set dimensions -- #\n")
        data_norm_n, data_norm_d = data_norm.shape
        log_value_to_file(log_file_train, data_norm_n, "data_norm_n")
        log_value_to_file(log_file_train, data_norm_d, "data_norm_d")

        # -- PCA the data -- #
        #log_file_train.write("# -- PCA the data -- #\n")
        n_pca = min(bits_to_encode_to, data_norm_d)
        cov_mat = np.cov(data_norm, rowvar=False)
        # log_array_to_file(log_file_train, cov_mat, "cov_mat")
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

        # -- Do eigenvectors check -- #
        for ev in eig_vecs:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        print('Everything ok, eigenvectors have all the same unit length 1!')

        # NB: signs of pc are slightly different than in matlab!!!! => same for the signs in data_norm_pcaed
        pc = np.fliplr(eig_vecs[:, -n_pca:])  # flip array in left-right direction
        data_norm_pcaed = data_norm.dot(pc)

        log_value_to_file(log_file_train, bits_to_encode_to, "bits_to_encode_to")
        #log_value_to_file(log_file_train, n_pca, "n_pca")
        # log_array_to_file(log_file_train, pc, "pc")


        # -- Fit uniform distribution -- #
        log_headline(log_file_train, "# -- Fit uniform distribution -- #")
        eps = np.finfo(float).eps
        mn = data_norm_pcaed.min(axis=0) - eps
        mx = data_norm_pcaed.max(axis=0) + eps
        log_value_to_file(log_file_train, mn, "mn")
        log_value_to_file(log_file_train, mx, "mx")

        # -- Enumerate eigenfunctions -- #
        log_headline(log_file_train, "# -- Enumerate eigenfunctions -- #")
        r = (mx - mn)
        norm_r = r/max(r)  #  + 0.000001
        max_mode = np.ceil((bits_to_encode_to + 1) * norm_r).astype(int)
        # log_value_to_file(log_file_train, max_mode, "max_mode")
        n_modes = np.sum(max_mode) - n_pca + 1
        # log_value_to_file(log_file_train, n_modes, "n_modes")
        modes = np.ones((int(n_modes), n_pca))
        # log_value_to_file(log_file_train, r, "r")
        # log_value_to_file(log_file_train, norm_r, "norm_r")

        m = 0
        for ith_pc in range(0, n_pca):
            row_index_to = m + max_mode[ith_pc]
            modes[(m + 1):row_index_to, ith_pc] = np.arange(2, max_mode[ith_pc] + 1)
            m = row_index_to - 1

        modes -= 1
        # log_array_to_file(log_file_train, modes, "modes")
        omega_zero = math.pi / r #  + 0.000001
        # log_value_to_file(log_file_train, omega_zero, "omega_zero")
        omegas = modes * np.tile(omega_zero, (n_modes, 1))
        # log_array_to_file(log_file_train, omegas, "omegas")

        eig_val = np.sum(np.power(omegas, 2), axis=1)
        log_array_to_file(log_file_train, sort(eig_val), "eig_val, but sorted")
        ii = eig_val.argsort()
        modes = modes[ii[1:bits_to_encode_to + 1], :]
        # log_array_to_file(log_file_train, modes, "final modes")


        # -- Store how many times each PC gets cut (basically how many bits each PC gets) -- #
        store_num_of_bits_each_pc_gets(modes, log_file_train)

        # sh_param = SHParam(mn, mx, modes, bits_to_encode_to)
        sh_model = SHModel(mn, mx, modes, bits_to_encode_to, pc, omega_zero, data_train_filename_location)

        # -- Time training phase -- #
        elapsed_time_formatted = time_process(start, time.time())
        log_value_to_file(log_file_train, elapsed_time_formatted, "elapsed time for training")

    # -- Close log training file -- #
    log_file_train.close()

    return sh_model


def compress_dataset(data_norm, sh_model):
    # -- Get dataset dimensions -- #
    data_norm_n, data_norm_d = data_norm.shape

    # -- PCA the given dataset according to the training set principal components -- #
    data_norm_pcaed = data_norm.dot(sh_model.pc_from_training)

    # -- Move towards the actual compression -- #
    data_norm_pcaed_and_centered = data_norm_pcaed - np.tile(sh_model.mn, (data_norm_n, 1))
    omegas_compress_training = sh_model.modes * np.tile(sh_model.omega_zero, (sh_model.n_bits, 1))
    u = np.zeros((data_norm_n, sh_model.n_bits), dtype=bool)
    # u = csc_matrix((data_norm_n, sh_param.n_bits), dtype=np.bool).todense()

    for ith_bit in range(0, sh_model.n_bits):
        omega_i = np.tile(omegas_compress_training[ith_bit, :], (data_norm_n, 1))
        ys = np.sin(data_norm_pcaed_and_centered * omega_i + math.pi / 2)
        yi = np.prod(ys, axis=1)
        u[:, ith_bit] = yi > 0

    u_compactly_binarized = compact_bit_matrix(u)
    return u, u_compactly_binarized


def sh():
    # -- Define args from terminal -- #
    args = init_args()

    # -- Define BIT_CNT_MAP -- #
    BIT_CNT_MAP = init_bitmap()

    # -- Define/Read params -- #
    average_number_neighbors = args.k
    loop_bits = [int(item) for item in args.loopbits.split(',')]
    data_filename = args.i
    max_hamming_distance_tested = args.mhd
    number_of_splits_testing_gt = args.nost
    log_file_train_destination = args.log_train
    log_file_test_destination = args.log_test #[TO DO logging evaluation metric]


    # -- Import datasets -- #
    delimiter = ' '
    data_train = np.genfromtxt(data_filename + '.train', delimiter=delimiter, dtype=np.float)
    data_test = np.genfromtxt(data_filename + '.test', delimiter=delimiter, dtype=np.float)
    print("# -- DONE READING -- #")

    # -- In case datasets are python 3D toy datasets (s_curve and ...), only pass the first 3 dimensions, since the last one is colors column -- #
    if ("s-curve" in data_filename) or ("swiss-roll" in data_filename):
        data_train = data_train[:, 0:-1]
        data_test = data_test[:, 0:-1]

    # -- Get datasets' sizes -- #
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]

    # -- Normalize data to unit hypercube -- #
    data_train_norm = normalize_data(data_train)
    data_test_norm = normalize_data(data_test)
    print("# -- DONE NORMALIZING -- #")

    # -- Define ground-truth neighbors(so all in Euclidean space) -- #
    d_ball_eucl, w_true_test_training = calculate_approximate_ground_truth_with_d_ball(data_train_norm, data_test_norm, average_number_neighbors, n_test, n_train, number_of_splits_testing_gt)

    # -- Demo Spectral Hashing -- #
    i = 0
    all_scores_precision_d_ball = []
    all_scores_recall_d_ball = []
    all_scores_f_measure_d_ball = []

    all_scores_reverse_indices = []
    for bits_to_encode_to in loop_bits:
        print("# -- START: SH for bits_to_encode = " + str(bits_to_encode_to) + " -- #")

        # -- Training is timed -- #
        sh_model = train_sh(data_train_norm, bits_to_encode_to, data_filename + '.train', log_file_train_destination)
        print("# -- DONE TRAINING -- #")

        # -- compressSH.m: For training set -- #
        u_training, u_compactly_binarized_training = compress_dataset(data_train_norm, sh_model)
        # check_buckets_balance_constraint(sh_model.n_bits, u_compactly_binarized_training, log_file_train_destination)

        print("# -- DONE COMPRESSION(TRAINIING) -- #")
        # print u_compactly_binarized_training

        # -- compressSH.m: For testing set -- #
        u_testing, u_compactly_binarized_testing = compress_dataset(data_test_norm, sh_model)
        print("# -- DONE COMPRESSION(TESTING) -- #")
        # print u_compactly_binarized_testing


        scores_reverse_indices = evaluate_with_reverse_indices(data_train_norm, data_test_norm, u_training, u_testing, average_number_neighbors)
        all_scores_reverse_indices.append(scores_reverse_indices)


        # -- Find out d_hamm_ball for u_training (for the bit vectors in Hamming) => basically the d_ball we calculated in Euclidean, but what it corresponds to in the mapped Hamming space -- #
        d_hamm_ball = calculate_d_ball(average_number_neighbors, 'hamming', u_training)
        print("d_hamm_ball", d_hamm_ball)
        max_hamming_distance_tested = int(np.ceil(d_hamm_ball))
        score_recall = np.zeros((max_hamming_distance_tested, 1))
        score_precision = np.zeros((max_hamming_distance_tested, 1))

        # -- Evaluate approximate GT precision && recall with d_ball(s) for the given bits_to_encode category -- #
        score_precision[:, 0], score_recall[:, 0] = evaluate_with_approximate_gt_d_balls(
                                                            w_true_test_training,
                                                            u_compactly_binarized_training,
                                                            u_compactly_binarized_testing,
                                                            max_hamming_distance_tested,
                                                            BIT_CNT_MAP)
        score_f_measure = calculate_f_score(score_precision, score_recall)
        print("# -- DONE PRECISION, RECALL and F_MEASURE(F1) -- #")
        all_scores_precision_d_ball.append(score_precision)
        all_scores_recall_d_ball.append(score_recall)
        all_scores_f_measure_d_ball.append(score_f_measure)

        i = i + 1
        print("# -- END: SH for bits_to_encode = " + str(bits_to_encode_to) + " -- #")

    # -- Print scores for approximate GT (with or without avg hamm_d_ball) -- #
    print_scores(
        np.hstack(all_scores_precision_d_ball),
        np.hstack(all_scores_recall_d_ball),
        np.hstack(all_scores_f_measure_d_ball))

    # -- Print scores for reverse indices GT -- #
    print(np.hstack(all_scores_reverse_indices))

def main():
    sh()

if __name__ == "__main__":
   main()




