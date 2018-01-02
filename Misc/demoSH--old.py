import numpy as np
import math
import random as rand
from scipy.spatial import distance
from shparams import SHParam
from distances import *

from scipy.sparse import *
from scipy import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input file name (same for both testing and training so far)", required=1)
parser.add_argument("-k", help="average number of nearest neighbours", type=int, required=1)
parser.add_argument("-mhd", help="maximum Hamming distance used for testing(in evaluation)", type=int, required=1)
args = parser.parse_args()

BIT_CNT_MAP = np.array([bin(i).count("1") for i in range(256)], np.uint16)


def normalize_data(data):
    data_norm = []
    for dimension in data.T:
        min_val_dim = min(dimension)
        max_val_dim = max(dimension)
        data_norm.append([(value - min_val_dim) / (max_val_dim - min_val_dim) for value in dimension])
    return np.array(data_norm).T


# -- All credit for this function goes to https://github.com/wanji/sh/blob/master/sh.py -- #
def compact_bit_matrix(bit_matrix):
    n_size, n_bits = bit_matrix.shape
    n_words = (n_bits + 7) / 8
    b = np.hstack([np.packbits(bit_matrix[:, i*8:(i+1)*8][:, ::-1], 1)
                   for i in range(int(n_words))])
    residue = n_bits % 8
    if residue != 0:
        b[:, -1] = np.right_shift(b[:, -1], 8 - residue)

    return b



# -- All credit for this function goes to https://github.com/wanji/sh/blob/master/sh.py -- #
def hammingDist(B1, B2):

    if B1.ndim == 1:
        B1 = B1.reshape((1, -1))

    if B2.ndim == 1:
        B2 = B2.reshape((1, -1))

    npt1, dim1 = B1.shape
    npt2, dim2 = B2.shape

    if dim1 != dim2:
        raise Exception("Dimension not consists: %d, %d" % (dim1, dim2))

    Dh = np.zeros((npt1, npt2), np.uint16)

    for i in range(npt1):
        Dh[i, :] = BIT_CNT_MAP[np.bitwise_xor(B1[i, :], B2)].sum(1)

    return Dh



def trainSH(data_norm, bits_to_encode):
    # -- Get training set dimensions -- #
    data_norm_n, data_norm_d = data_norm.shape

    # -- PCA the data -- #
    n_pca = min(bits_to_encode, data_norm_d)
    cov_mat = np.cov(data_norm, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    # NB: signs of pc are slightly different than in matlab!!!! => same for the signs in data_norm_pcaed
    pc = np.fliplr(eig_vecs[:, -n_pca:])  # flip array in left-right direction
    data_norm_pcaed = data_norm.dot(pc)

    # -- Fit uniform distribution -- #
    eps = np.finfo(float).eps
    mn = data_norm_pcaed.min(axis=0) - eps
    mx = data_norm_pcaed.max(axis=0) + eps

    # -- Enumerate eigenfunctions -- #
    r = (mx - mn)
    norm_r = r/max(r)
    max_mode = np.ceil((bits_to_encode + 1) * norm_r).astype(int)
    # print max_mode
    n_modes = np.sum(max_mode) - n_pca + 1
    modes = np.ones((int(n_modes), n_pca))

    m = 0
    for ith_pc in range(0, n_pca):
        row_index_to = m + max_mode[ith_pc]
        modes[(m + 1):row_index_to, ith_pc] = np.arange(2, max_mode[ith_pc] + 1)
        m = row_index_to - 1

    modes -= 1
    omega_zero = math.pi / r
    omegas = modes * np.tile(omega_zero, (n_modes, 1))
    eig_val = np.sum(np.power(omegas, 2), axis=1)
    ii = eig_val.argsort()
    modes = modes[ii[1:bits_to_encode + 1], :]

    sh_param = SHParam(mn, mx, modes, bits_to_encode)

    return sh_param, pc, omega_zero


def compress_dataset(data_norm, pc_from_training, sh_param, omega_zero):
    # -- Get dataset dimensions -- #
    data_norm_n, data_norm_d = data_norm.shape

    # -- PCA the given dataset according to the training set -- #
    data_norm_pcaed = data_norm.dot(pc_from_training)

    # -- Move towards the actual compression -- #
    data_norm_pcaed_and_centered = data_norm_pcaed - np.tile(sh_param.mn, (data_norm_n, 1))
    omegas_compress_training = sh_param.modes * np.tile(omega_zero, (sh_param.n_bits, 1))
    u = np.zeros((data_norm_n, sh_param.n_bits), dtype=bool)
    # u = csc_matrix((data_norm_n, sh_param.n_bits), dtype=np.bool).todense()
    # print("u[0]", u[0])
    # print(u.shape)

    for ith_bit in range(0, sh_param.n_bits):
        omega_i = np.tile(omegas_compress_training[ith_bit, :], (data_norm_n, 1))
        ys = np.sin(data_norm_pcaed_and_centered * omega_i + math.pi / 2)
        yi = np.prod(ys, axis=1)
        # print(yi, yi.shape)
        # print(u[:, ith_bit], u[:, ith_bit].shape)
        u[:, ith_bit] = yi > 0

    u_compactly_binarized = compact_bit_matrix(u)
    return u, u_compactly_binarized



def calculate_precision_and_recall(w_true_test_training, u_compactly_binarized_training, u_compactly_binarized_testing, max_hamming_distance_tested):
    total_good_pairs = w_true_test_training.sum()
    retrieved_good_pairs = np.zeros((max_hamming_distance_tested, 1))
    retrieved_pairs = np.zeros((max_hamming_distance_tested, 1))

    score_precision = np.zeros((max_hamming_distance_tested, 1))
    score_recall = np.zeros((max_hamming_distance_tested, 1))

    print("total_good_pairs", total_good_pairs)

    # -- We do hamming dist. from each testing point to all training points -- #
    for ith_testing, compact_testing_point in enumerate(u_compactly_binarized_testing):
        distances_from_testing_to_all_training = BIT_CNT_MAP[np.bitwise_xor(compact_testing_point, u_compactly_binarized_training)].sum(1)
        for hamming_distance_used_to_test_against in range(0, max_hamming_distance_tested):
            indices_pairs_of_good_pairs_in_d_hamm = np.where(distances_from_testing_to_all_training < hamming_distance_used_to_test_against + 0.00001)

            retrieved_good_pairs[hamming_distance_used_to_test_against][0] += sum(
                [w_true_test_training[ith_testing, pair_index] for pair_index in indices_pairs_of_good_pairs_in_d_hamm])
            retrieved_pairs[hamming_distance_used_to_test_against][0] += size(indices_pairs_of_good_pairs_in_d_hamm)

    for hamming_distance_used_to_test_against in range(0, max_hamming_distance_tested):
        score_precision[hamming_distance_used_to_test_against][0] = retrieved_good_pairs[hamming_distance_used_to_test_against][0] / (retrieved_pairs[hamming_distance_used_to_test_against][0] * 1.0)
        score_recall[hamming_distance_used_to_test_against][0] = retrieved_good_pairs[hamming_distance_used_to_test_against][0] / (total_good_pairs * 1.0)
    print(score_precision)
    print(score_recall)
    return score_precision.T, score_recall.T


def calculate_ground_truth(data_train_norm, data_test_norm, average_number_neighbors, n_test, n_train):
    d_true_training = distance.cdist(data_train_norm, data_train_norm, 'euclidean')
    d_true_training.sort(axis=1)
    d_ball = np.mean(d_true_training[:, average_number_neighbors - 1])
    print("d_ball", d_ball)

    number_of_splits_testing = 10
    w_true_test_training = csc_matrix((n_test, n_train), dtype=np.bool).todense()
    data_test_norm_chunks = np.array_split(data_test_norm, number_of_splits_testing)
    test_chunk_size = data_test_norm_chunks[0].shape[0]
    # print(test_chunk_size)  # test_chunk_size * test_chunk_ith + 0/1/2
    for test_chunk_ith, test_chunk in enumerate(data_test_norm_chunks):
        d_true_test_training_chunk = distance.cdist(test_chunk, data_train_norm, 'euclidean')
        w_true_test_training[test_chunk_size * test_chunk_ith:test_chunk_size * test_chunk_ith + test_chunk_size,
        :] = d_true_test_training_chunk < d_ball
    print(w_true_test_training)
    print(w_true_test_training.shape)

    return w_true_test_training




def sh():
    # Define params
    average_number_neighbors = args.k
    loop_bits = [2]
    data_filename = args.i
    max_hamming_distance_tested = args.mhd
    folder_path = './Data/Profi/'

    # Import datasets
    delimiter = ' '
    data_train = np.genfromtxt(folder_path + data_filename + '.train', delimiter=delimiter, dtype=np.float)
    data_test = np.genfromtxt(folder_path + data_filename + '.test', delimiter=delimiter, dtype=np.float)
    print("# -- DONE READING -- #")

    # Get datasets' sizes
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]

    # Normalize data to unit hypercube
    data_train_norm = normalize_data(data_train)
    data_test_norm = normalize_data(data_test)
    print("# -- DONE NORMALIZING -- #")

    # data_train_norm = data_train
    # data_test_norm = data_test

    # Define ground-truth neighbors (used for evaluation)
    # -- GT, version 2(GOOD so far): batching only testing -- #
    w_true_test_training = calculate_ground_truth(data_train_norm, data_test_norm, average_number_neighbors, n_test, n_train)


    # calculate_ground_truth_as_matlab(data_train_norm, data_test_norm, average_number_neighbors)
    # TO DO: try to implement distMat from Matlab

    # Demo Spectral Hashing
    i = 0
    score_recall = np.zeros((max_hamming_distance_tested, len(loop_bits)))
    score_precision = np.zeros((max_hamming_distance_tested, len(loop_bits)))
    for bits_to_encode in loop_bits:
        print("# -- START: SH for bits_to_encode = " + str(bits_to_encode) + " -- #")

        sh_param, pc_from_training, omega_zero = trainSH(data_train_norm, bits_to_encode)
        print("# -- DONE TRAINING -- #")


        # -- compressSH.m: For training set -- #
        u_training, u_compactly_binarized_training = compress_dataset(data_train_norm, pc_from_training, sh_param, omega_zero)
        print("# -- DONE COMPRESSION(TRAINIING) -- #")
        # print u_compactly_binarized_training

        # -- compressSH.m: For testing set -- #
        u_testing, u_compactly_binarized_testing = compress_dataset(data_test_norm, pc_from_training, sh_param, omega_zero)
        print("# -- DONE COMPRESSION(TESTING) -- #")
        # print u_compactly_binarized_testing

        # -- Calculate Hamming distance between compacted testing set and compacted training set -- #
        # d_hamm = sh_param.n_bits * distance.cdist(u_testing, u_training, 'hamming')  # np.array_split(u_testing, 2)[0]
        # d_hamm = hammingDist(u_compactly_binarized_testing, u_compactly_binarized_training)

        # bla = BIT_CNT_MAP[np.bitwise_xor(u_compactly_binarized_training[0, :], u_compactly_binarized_testing)].sum(1)
        # print(bla)
        # print(u_compactly_binarized_training[0, :])
        # print(u_compactly_binarized_testing[0:10, :])

        # print("d_hamm[0]", d_hamm[0][3])
        # print("u_testing[0]", u_testing[0])
        # print("u_training[0:10, :]", u_training[0:10, :])
        # print("# -- DONE D_HAMM(TESTING, TRAINING) -- #")

        # -- Evaluate precision && recall for the given bits_to_encode category -- #
        score_precision[:, i], score_recall[:, i] = calculate_precision_and_recall(w_true_test_training, u_compactly_binarized_training, u_compactly_binarized_testing, max_hamming_distance_tested)
        print("# -- DONE PRECISION AND RECALL -- #")
        # score_precision[:, i], score_recall[:, i] = evaluate_for_bits_category(w_true_test_training, d_hamm, max_hamming_distance_tested)

        i = i + 1
        print("# -- END: SH for bits_to_encode = " + str(bits_to_encode) + " -- #")
    # print(score_precision)
    # print(score_recall)

def main():
    sh()

main()
