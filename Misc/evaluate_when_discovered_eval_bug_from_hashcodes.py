import datetime
import numpy as np
import math
import random as rand
import argparse
from scipy.spatial import distance
from scipy.sparse import *
from scipy import *
from sklearn.neighbors import NearestNeighbors
from shparams import SHParam
from shparams import SHModel

def print_all_good_nn_indices_in_hamming(indices, hamm_dist_debug, text):
    counter = 0
    for ind, indices in enumerate(
            indices[hamm_dist_debug]):
        print("{0}th, #{1}: {2}".format(ind, len(indices), indices))
        counter += len(indices)
    print("\nFinal counter check for #{0} => {1}\n\n".format(text, counter))


# -- Evaluation 1: with Euclidean d_ball and hamm_d_ball in range(0, max_hamming_distance_tested) -- #
def evaluate_with_approximate_gt_d_balls(w_true_test_training, u_compactly_binarized_training, u_compactly_binarized_testing, u_training, u_testing,  max_hamming_distance_tested, BIT_CNT_MAP, eval_debug_object, debug_mode=False):
    # print("w_true_test_training => \n {0}".format(w_true_test_training.astype(int)))
    total_good_pairs = w_true_test_training.sum()
    retrieved_good_pairs = np.zeros((max_hamming_distance_tested, 1))
    retrieved_pairs = np.zeros((max_hamming_distance_tested, 1))

    score_precision = np.zeros((max_hamming_distance_tested, 1))
    score_recall = np.zeros((max_hamming_distance_tested, 1))


    # print("score_recall", score_recall, max_hamming_distance_tested)
    # print("total_good_pairs", total_good_pairs)

    # -- We do hamming dist. from each testing point to all training points -- #
    # -- Start: HELPER DEBUG -- #
    if debug_mode:
        indices_pairs_of_good_pairs_in_d_hamm_for_all_queries = {k: [] for k in range(0, max_hamming_distance_tested)}
        indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl = {k: [] for k in range(0, max_hamming_distance_tested)}
    # -- End: HELPER DEBUG -- #
    for ith_testing, compact_testing_point in enumerate(u_compactly_binarized_testing):
        distances_from_testing_to_all_training = BIT_CNT_MAP[np.bitwise_xor(compact_testing_point, u_compactly_binarized_training)].sum(1)
        # print(distances_from_testing_to_all_training)
        # print("distances_from_testing_to_all_training, from test_point[{0}] => {1} \n".format(ith_testing, distances_from_testing_to_all_training))
        for hamming_distance_used_to_test_against in range(0, max_hamming_distance_tested):
            indices_pairs_of_good_pairs_in_d_hamm = np.where(distances_from_testing_to_all_training < hamming_distance_used_to_test_against + 0.00001)



            # -- Start: HELPER DEBUG -- #
            if debug_mode:
                # print(indices_pairs_of_good_pairs_in_d_hamm)
                gt_nn_indices_ith_testing = [idx for idx, nn_true in enumerate(w_true_test_training.astype(int)[ith_testing, :].tolist()[0]) if nn_true]
                validated_nn_indices_ith_testing = np.intersect1d(indices_pairs_of_good_pairs_in_d_hamm, gt_nn_indices_ith_testing)
                # print("validated_nn_indices_ith_testing")
                # print(validated_nn_indices_ith_testing, type(validated_nn_indices_ith_testing), validated_nn_indices_ith_testing[0])

                indices_pairs_of_good_pairs_in_d_hamm_for_all_queries[hamming_distance_used_to_test_against].append(list(indices_pairs_of_good_pairs_in_d_hamm[0]))
                indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl[hamming_distance_used_to_test_against].append(list(validated_nn_indices_ith_testing))
            # -- End: HELPER DEBUG -- #

            retrieved_good_pairs[hamming_distance_used_to_test_against][0] += sum(
                [w_true_test_training[ith_testing, pair_index] for pair_index in indices_pairs_of_good_pairs_in_d_hamm])
            retrieved_pairs[hamming_distance_used_to_test_against][0] += size(indices_pairs_of_good_pairs_in_d_hamm)

    # -- Start: HELPER DEBUG -- #
    if debug_mode:
        hamm_dist_debug = 0
        eval_debug_object.__set_hamm_dist_debug__(hamm_dist_debug)
        print("\nhamm_ball we search in is {0} => \n".format(hamm_dist_debug))

        # 0. nn indices for all the good pairs in Euclidean
        print("\n# -- nn indices for all the good pairs in Euclidean, plus counts => -- #")
        gt_nn_indices = []
        for tth, test_training_true_nn in enumerate(w_true_test_training):
            true_nn_indices = [vth for vth in range(0, w_true_test_training.shape[1]) if test_training_true_nn[0, vth]]
            gt_nn_indices.append(true_nn_indices)
            print("{0}th, #{1}: {2}".format(tth, len(true_nn_indices), true_nn_indices))

        # 1. nn indices for all the good pairs in Hamming, which also match with the ones in Euclidean (are 'validated' by the GT)
        print("\n# -- nn indices for all the good pairs in Hamming, which also match with the ones in Euclidean (are 'validated' by the GT), plus counts => -- #")
        print_all_good_nn_indices_in_hamming(indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl,
                                             hamm_dist_debug, "Indices of Euclidean-VALIDATED good neighbors")

        # 2. nn indices for all the good pairs in Hamming
        print("\n# -- nn indices for all the good pairs in Hamming, plus counts => -- #")
        print_all_good_nn_indices_in_hamming(indices_pairs_of_good_pairs_in_d_hamm_for_all_queries,
                                             hamm_dist_debug, "Indices of good neighbors")


        eval_debug_object.__set_indices__(gt_nn_indices, indices_pairs_of_good_pairs_in_d_hamm_for_all_queries_validated_by_eucl[hamm_dist_debug], indices_pairs_of_good_pairs_in_d_hamm_for_all_queries[hamm_dist_debug])
        # -- End: HELPER DEBUG -- #

    for hamming_distance_used_to_test_against in range(0, max_hamming_distance_tested):
        # # -- Start: HELPER DEBUG -- #
        # if hamming_distance_used_to_test_against == hamm_dist_debug - 1:
        # -- End: HELPER DEBUG -- #
        if debug_mode:
            if hamming_distance_used_to_test_against == hamm_dist_debug:
                print("\n\nP = retrieved_good_pairs / retrieved_pairs")
                print("R = retrieved_good_pairs / total_good_pairs")
                print("hamming_distance => {0}".format(hamming_distance_used_to_test_against))
                print(
                    "retrieved_good_pairs => {0}".format(retrieved_good_pairs[hamming_distance_used_to_test_against][0]))
                print("retrieved_pairs => {0}".format(retrieved_pairs[hamming_distance_used_to_test_against][0]))
                print("total_good_pairs => {0} \n\n".format(total_good_pairs))
                eval_debug_object.__set_indices_totals__(retrieved_good_pairs[hamming_distance_used_to_test_against][0], retrieved_pairs[hamming_distance_used_to_test_against][0], total_good_pairs)
        # else:
        #     print("# ----------------------------------------------------------- #")
        #     print("P = retrieved_good_pairs / retrieved_pairs \n")
        #     print("R = retrieved_good_pairs / total_good_pairs \n")
        #     print("hamming_distance => {0} \n".format(hamming_distance_used_to_test_against))
        #     print("retrieved_good_pairs => {0} \n".format(retrieved_good_pairs[hamming_distance_used_to_test_against][0]))
        #     print("retrieved_pairs => {0} \n".format(retrieved_pairs[hamming_distance_used_to_test_against][0]))
        #     print("total_good_pairs => {0} \n".format(total_good_pairs))
        score_precision[hamming_distance_used_to_test_against][0] = retrieved_good_pairs[hamming_distance_used_to_test_against][0] / (retrieved_pairs[hamming_distance_used_to_test_against][0] * 1.0 + 0.00000001)
        score_recall[hamming_distance_used_to_test_against][0] = retrieved_good_pairs[hamming_distance_used_to_test_against][0] / (total_good_pairs * 1.0 + 0.00000001)

    return score_precision.T, score_recall.T, eval_debug_object


def calculate_d_ball(average_number_neighbors, metric, data_norm):  # data_norm is usually data_train_norm in our case.
    nbrs = NearestNeighbors(n_neighbors=average_number_neighbors, algorithm='auto', metric=metric).fit(
        data_norm)
    distances, indices = nbrs.kneighbors(data_norm)
    # print("\n# -- Step calculate_d_ball, where distances are => -- #")
    # print(distances)
    d_ball = np.mean(distances[:, average_number_neighbors - 1])
    print("\n# -- d_ball = {0} -- #".format(d_ball))
    return d_ball

def calculate_approximate_ground_truth_with_d_ball(data_train_norm, data_test_norm, average_number_neighbors, n_test, n_train, number_of_splits_testing_gt):
    # -- Define d_ball from the training set -- #
    d_ball = calculate_d_ball(average_number_neighbors, 'euclidean', data_train_norm)
    print("\nPASSED d_ball calculation => dball = {0}\n".format(d_ball))

    # -- Define, for each testing point, which of the points in the training set is within its d_ball radius -- #
    w_true_test_training = csc_matrix((n_test, n_train), dtype=np.bool).todense()
    data_test_norm_chunks = np.array_split(data_test_norm, number_of_splits_testing_gt)
    # test_chunk_size = data_test_norm_chunks[0].shape[0]
    # print(test_chunk_size)  # test_chunk_size * test_chunk_ith + 0/1/2
    for test_chunk_ith, test_chunk in enumerate(data_test_norm_chunks):
        d_true_test_training_chunk = distance.cdist(test_chunk, data_train_norm, 'euclidean')
        count_index_extension = d_true_test_training_chunk.shape[0]
        w_true_test_training[count_index_extension * test_chunk_ith:count_index_extension * test_chunk_ith + count_index_extension,
        :] = d_true_test_training_chunk < d_ball

    print("w_true_test_training.shape", w_true_test_training.shape)


    return d_ball, w_true_test_training



# -- Evaluation 2: Exact GT, with reverse indices -- #
def get_reverse_indices(euclidean_indices, hamming_indices, k, query_size):
    all_queries_reverse_indices = np.zeros((query_size, k), dtype=int)
    for sq_ei_ith, single_query_euclidean_indices in enumerate(euclidean_indices):
        sq_reverse_indices = [hamming_indices[sq_ei_ith].tolist().index(eucl_index) for eucl_index_jth, eucl_index in
                              enumerate(single_query_euclidean_indices)]
        all_queries_reverse_indices[sq_ei_ith] = sq_reverse_indices
    return all_queries_reverse_indices


def evaluate_with_reverse_indices(data_train_norm, data_test_norm, u_training, u_testing, k):
    # Obs 1 => Euclidean space: query_set = data_test_norm, set_we_look_up = data_train_norm
    # Obs 2 => Hamming space: query_set = u_testing, set_we_look_up = u_training

    # -- Define vars -- #
    searching_space_size = data_train_norm.shape[0]
    query_size = data_test_norm.shape[0]
    num_of_batches = int(np.ceil(query_size / 10))

    # -- Split the query set -- #
    query_batches_hamming = array_split(u_testing, num_of_batches)
    query_batches_euclidean = array_split(data_test_norm, num_of_batches)

    # -- Find GT in Euclidean space -- #
    euclidean_nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(data_train_norm)
    # -- Find GT in Hamming space -- #
    hamming_nbrs = NearestNeighbors(n_neighbors=searching_space_size, algorithm='auto', metric='hamming').fit(u_training)

    # -- Make scale of retrieved samples needed for recall calculation -- #
    freqs = range(1, 1 + int(math.log2(searching_space_size)))  #  not so sure about this
    retrieved_points_scale = [np.power(2, x) for x in freqs]
    retrieved_points_scale.append(k)

    # -- Keep in mind that last row for results represents precision for the #bits we test and the provided k -- #
    results = np.zeros((len(retrieved_points_scale), 2))
    results[:, 0] = retrieved_points_scale

    for b_th, query_batch_euclidean in enumerate(query_batches_euclidean):
        euclidean_indices = euclidean_nbrs.kneighbors(query_batches_euclidean[b_th], return_distance=False)
        hamming_indices = hamming_nbrs.kneighbors(query_batches_hamming[b_th], return_distance=False)

        # -- Calculate reverse indices matrix -- #
        reverse_indices = get_reverse_indices(euclidean_indices, hamming_indices, k, len(query_batch_euclidean))

        for rp_th, retrieved_points in enumerate(retrieved_points_scale):
            # -- Calculate recall for this specific value of retrieved_points -- #
            true_positives = 0.0
            for qp_ri, query_point_reverse_indices in enumerate(reverse_indices):
                true_positives += sum((reverse_index < retrieved_points) for ri, reverse_index in enumerate(query_point_reverse_indices))

            # recall = true_positives / (query_size * 1.0 * k)
            results[rp_th, 1] += true_positives


    results[:, 1] /= query_size * 1.0 * k

    # Obs 3 => Last row in results corresponds to value for precision, for the selected #bits.
    return results