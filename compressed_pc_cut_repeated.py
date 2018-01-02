from helpers import *
import operator

def compress_dataset__pc_cut_repeated(data_train_norm, data_test_norm, sh_model, dataset_label):
    print("\n# -- BALANCED, but with PC CUT REPEATED BIT EFFECT, where PCs are cut multiple times: {0} set -- #".format(dataset_label))
    # -- Define some params -- #
    corner_case_num_buckets = sh_model.n_bits  # * 128

    # -- Get dataset dimensions -- #
    data_train_norm_n, data_train_norm_d = data_train_norm.shape
    data_test_norm_n, data_test_norm_d = data_test_norm.shape

    # -- PCA the given dataset according to the training set principal components -- #
    data_train_norm_pcaed = data_train_norm.dot(sh_model.pc_from_training)
    data_test_norm_pcaed = data_test_norm.dot(sh_model.pc_from_training)

    # -- Move towards the actual compression -- #
    data_train_norm_pcaed_and_centered = data_train_norm_pcaed - np.tile(sh_model.mn, (data_train_norm_n, 1))
    data_test_norm_pcaed_and_centered = data_test_norm_pcaed - np.tile(sh_model.mn, (data_test_norm_n, 1))

    # -- This is for 'uord', but I think 'ord' will be excluded eventually -- #
    pcs_to_loop_through = enumerate(range(0, sh_model.n_bits))

    # -- Get data boxes -- #
    data_box_train = get_sine_data_box(data_train_norm_pcaed_and_centered, sh_model, data_train_norm_n)
    data_box_test = get_sine_data_box(data_test_norm_pcaed_and_centered, sh_model, data_test_norm_n)

    # -- Find out how many bits each pc contributes with -- #
    pcs_ith_bits_mapping, pcs_ith_bits_when_multiple_cuts, first_pcs_when_axis_cut_multiple_times = get_pcs_ith_bits_mapping(sh_model)
    pcs_we_store_bits_for = [tup[1] for tth, tup in enumerate(pcs_ith_bits_when_multiple_cuts)]

    if dataset_label == "testing":
        print_help("pcs_we_store_bits_for", pcs_we_store_bits_for)
        print("\npcs_ith_bits_mapping: {0}".format(pcs_ith_bits_mapping))
        print("\nsh_model.modes.shape: {0}".format(sh_model.modes.shape))
        print("\npcs_to_loop_through: {0}".format(range(0, sh_model.n_bits)))
        print("\npcs_ith_bits_when_multiple_cuts: {0}".format(pcs_ith_bits_when_multiple_cuts))
        print("\nfirst_pcs_when_axis_cut_multiple_times: {0}".format(first_pcs_when_axis_cut_multiple_times))

    if dataset_label == "training":
        data_hashcodes = [[] for _ in range(0, data_train_norm_n)]
    else:
        data_hashcodes = [[] for _ in range(0, data_test_norm_n)]

    grey_codes_per_pc = {}
    for pth, pc in pcs_to_loop_through:
        if str(pc) in pcs_ith_bits_mapping.keys(): # quick fix. must be investigated
            num_bits_of_contribution = len(pcs_ith_bits_mapping[str(pc)]) if len(pcs_ith_bits_mapping[str(pc)]) > 0 else 1 # this is not the idea until the end and can confuse!
            num_buckets_per_pc = corner_case_num_buckets if np.power(2, num_bits_of_contribution) == 0 else np.power(2, num_bits_of_contribution)

            # -- Establish box information based only on training -- #
            pc_scores_train, max_pc_score_train, min_pc_score_train, range_pc_train, interval_pc_train = get_data_box_info(data_box_train, pc, num_buckets_per_pc)

            # -- GreyCode stuff -- #
            gray_codes_pc = list(GrayCode(num_bits_of_contribution).generate_gray())
            grey_codes_per_pc[str(pc)] = gray_codes_pc
            num_gray_codes = len(gray_codes_pc)

            # -- Establish pc scores/actual scores/data box scores which is about to be partitioned -- #
            pc_scores = data_box_train[:, pc] if dataset_label == "training" else data_box_test[:, pc]

            # -- Go through each data point in my pc box and check only the score corresponding to that pc dimension -- #
            for dp, pc_score in enumerate(pc_scores):
                if len(pcs_ith_bits_mapping[str(pc)]) > 0:
                    bucket_index = int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train)))
                    bucket_index = 0 if bucket_index < 0 else bucket_index
                    provenance_bucket_index = bucket_index if bucket_index < num_gray_codes else num_gray_codes - 1
                    bits_to_attach = [int(bit_str) for bit_str in gray_codes_pc[provenance_bucket_index]]

                    # if dataset_label == "testing" and dp == 49:
                    #     print("data_box:\n{0}".format(pc_scores))
                    #     print("ith_bit={0}\n{1}th PC, with pc_cols:{4}\npc_score={2}\nbit_assigned={3}\n----------------------------------------------".format(pth, pc, pc_score, gray_codes_pc[provenance_bucket_index], pcs_ith_bits_mapping[str(pc)]))  # data_box={4} / pc_scores
                    #     print("")


                    data_hashcodes[dp] = np.hstack((data_hashcodes[dp], bits_to_attach))

    u = np.array(data_hashcodes, dtype=bool)
    u_compactly_binarized = compact_bit_matrix(u)

    return u, u_compactly_binarized






#
# # -- MUST BE BASED/DOING THE SAME AS THE FOLLOWING BUGGED VERSION -- #
#
# # -- Balanced SH Compression: NB => This version contains a doubling 'bug', which contributes with the following: 1 - it doubles come bits, for PCs cut more than 1 time, 2 - it obviously ignores PCs which don't give any bits at all -- #
# def compress_dataset__balanced_partitioning(data_train_norm, data_test_norm, sh_model, dataset_label, condition_ordered_pcs):
#     print("\n# -- BALANCED: {0} set -- #".format(dataset_label))
#     # -- Define some params -- #
#     corner_case_num_buckets = sh_model.n_bits  # * 128
#
#     # -- Get dataset dimensions -- #
#     data_train_norm_n, data_train_norm_d = data_train_norm.shape
#     data_test_norm_n, data_test_norm_d = data_test_norm.shape
#
#     # -- PCA the given dataset according to the training set principal components -- #
#     data_train_norm_pcaed = data_train_norm.dot(sh_model.pc_from_training)
#     data_test_norm_pcaed = data_test_norm.dot(sh_model.pc_from_training)
#
#     # -- Move towards the actual compression -- #
#     data_train_norm_pcaed_and_centered = data_train_norm_pcaed - np.tile(sh_model.mn, (data_train_norm_n, 1))
#     data_test_norm_pcaed_and_centered = data_test_norm_pcaed - np.tile(sh_model.mn, (data_test_norm_n, 1))
#
#     # -- ORDER the modes if condition_ordered_pcs -- #
#     if condition_ordered_pcs == "ord":
#         sh_model.modes = order_modes_for_bit_contribution(sh_model)
#         pcs_ordered = get_pc_order_as_vanilla(sh_model)
#         pcs_to_loop_through = enumerate(pcs_ordered)
#         print_help("\npcs_ordered", pcs_ordered)
#     else:
#         pcs_to_loop_through = enumerate(range(0, sh_model.n_bits))
#
#
#     # -- Create data box similarly to compress_dataset, the only upcoming difference being the partitioning with equally spaced-out line cuts instead of sine curves cuts -- #
#     data_box_train = get_sine_data_box(data_train_norm_pcaed_and_centered, sh_model, data_train_norm_n)
#     data_box_test = get_sine_data_box(data_test_norm_pcaed_and_centered, sh_model, data_test_norm_n)
#
#     # -- Find out how many bits each pc contributes with -- #
#     bits_per_pcs = get_pc_bitwise_contribution(sh_model)
#     print("\nbits_per_pcs: {0}".format(bits_per_pcs))
#
#     if dataset_label == "training":
#         data_hashcodes = [[] for _ in range(0, data_train_norm_n)]
#     else:
#         data_hashcodes = [[] for _ in range(0, data_test_norm_n)]
#
#
#     grey_codes_per_pc = {}
#     # for pc in range(0, sh_model.n_bits):
#     for pth, pc in pcs_to_loop_through:
#         if str(pc) in bits_per_pcs.keys():
#             num_bits_of_contribution = len(bits_per_pcs[str(pc)])
#
#             # -- TRY TO GET THE ACTUAL PC to PARTITION, in case we get bits from pcs_ordered, because otherwise pc variable would be off (not really stand for the PC we actually want to get bits from) -- #
#             if condition_ordered_pcs == "ord":
#                 pc = get_next_pc_to_attach_bits_from(bits_per_pcs, pcs_ordered, pc)
#
#             # Obs(*): In case num_buckets_per_pc, it means we have way too many buckets and their int value overflows
#             num_buckets_per_pc = corner_case_num_buckets if np.power(2, num_bits_of_contribution) == 0 else np.power(2, num_bits_of_contribution)
#
#             # -- Establish box information based only on training -- #
#             pc_scores_train = data_box_train[:, pc]
#             max_pc_score_train = max(pc_scores_train)
#             min_pc_score_train = min(pc_scores_train)
#             range_pc_train = max_pc_score_train - min_pc_score_train
#             interval_pc_train = range_pc_train / (num_buckets_per_pc * 1.0)
#
#             gray_codes_pc = list(GrayCode(num_bits_of_contribution).generate_gray())
#             grey_codes_per_pc[str(pc)] = gray_codes_pc
#             num_gray_codes = len(gray_codes_pc)
#
#             # -- Plot box/PC partitions, but ALWAYS determining cut_points for training, no matter if we compress training or testing -- #
#             cut_points_train = [min_pc_score_train + interval_pc_train * ith for ith in range(1, num_gray_codes)]
#
#
#             # -- Establish pc scores/actual scores/data box scores which is about to be partitioned -- #
#             if dataset_label == "training":
#                 pc_scores = data_box_train[:, pc]
#             else:
#                 pc_scores = data_box_test[:, pc]
#
#             # -- Check how points are distributed per each pc -- #
#             points_distrib_per_pc = [0 for _ in range(0, len(gray_codes_pc))]
#
#             # -- Go through each data point in my pc box and check only the score corresponding to that pc dimension -- #
#             for dp, pc_score in enumerate(pc_scores):
#
#
#                 bucket_index = int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train)))
#                 if bucket_index < 0:
#                     bucket_index = 0
#                 provenance_bucket_index = bucket_index if bucket_index < num_gray_codes else num_gray_codes - 1
#                 provenance_bucket_index_v2 = bucket_index % num_gray_codes
#
#                 if dataset_label == "testing" and dp == 49:
#                     print("ith_bit={0}, {1}th PC, pc_score={2}, bit_assigned={3}, => ".format(pth, pc, pc_score, gray_codes_pc[provenance_bucket_index])) # data_box={4} / pc_scores
#                     print("")
#
#                 # if dataset_label == "testing" and dp == 47:
#                 #     print
#                 # ("{0}th PC, pc_score={1}, bit_assigned={2}".format(pc, pc_score, gray_codes_pc[provenance_bucket_index]))
#
#                 data_hashcodes[dp] = np.hstack((data_hashcodes[dp], [int(bit_str) for bit_str in gray_codes_pc[provenance_bucket_index]]))
#
#                 # if dataset_label == "testing" and dp == 47 and (pc == 8):
#                 #     print("HASHCODE after appending bits for 0th PC, data_hashcodes[dp]={0}, total_len={1}".format(data_hashcodes[dp], len(data_hashcodes[dp])))
#
#                 # print("pc_" + str(pc) + ", point_" + str(dp) + ", hashcode_" + str(data_hashcodes[dp]) + ", bucket-index_" + str(provenance_bucket_index) + ", bucket-index-unrounded_" + str((pc_score - min_pc_score) / (interval_pc)))
#
#                 points_distrib_per_pc[provenance_bucket_index] += 1
#
#     u = np.array(data_hashcodes, dtype=bool)
#     u_compactly_binarized = compact_bit_matrix(u)
#
#     print("u[49]", u[49].astype(int))
#
#     return u, u_compactly_binarized
