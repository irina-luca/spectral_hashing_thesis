# from helpers import *
# import operator
#
#
#
#
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
#     # if dataset_label == "testing":
#     #     np.set_printoptions(threshold=np.nan)
#     #     print("data_box_test => ", data_box_test)
#
#     # print_min_max_data_box(data_box_train, "data_box_train")
#     # print_min_max_data_box(data_box_test, "data_box_test")
#
#
#     # -- Find out how many bits each pc contributes with -- #
#     bits_per_pcs = get_pc_bitwise_contribution(sh_model)
#     # print("\nbits_per_pcs: {0}".format(bits_per_pcs))
#
#     # ordered_keys, bits_contribution_and_ordering_per_pcs = get_pc_bitwise_contribution_and_ordering(sh_model)
#     # print("\nbits_contribution_and_ordering_per_pcs: {0}".format(bits_contribution_and_ordering_per_pcs))
#     # print("\nordered_keys: {0}".format(ordered_keys))
#
#     if dataset_label == "training":
#         data_hashcodes = [[] for _ in range(0, data_train_norm_n)]
#     else:
#         data_hashcodes = [[] for _ in range(0, data_test_norm_n)]
#
#
#
#
#     grey_codes_per_pc = {}
#     # for pc in range(0, sh_model.n_bits):
#     for pth, pc in pcs_to_loop_through:
#         # -- Check if the Principal Component pc contributes to our hashcode generation at all -- #
#         if str(pc) in bits_per_pcs.keys():  # This is actually redundant if pcs_to_loop_through are pcs_ordered, but I leave it here anyways
#             # -- Establish n_buckets and n_bits -- #
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
#             # plot_2D(np.hstack((np.vstack(pc_scores), np.zeros((len(pc_scores), 1)))), True, "Balanced", cut_points_train, "orange")
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
#                 if dataset_label == "testing" and dp == 47:
#                     print("ith_bit={0}, {1}th PC, pc_score={2}, bit_assigned={3}, data_box={4} => ".format(pth, pc, pc_score, gray_codes_pc[provenance_bucket_index], pc_scores))
#                     print("")
#
#                 # if dataset_label == "testing" and dp == 47:
#                 #     print
#                 # ("{0}th PC, pc_score={1}, bit_assigned={2}".format(pc, pc_score, gray_codes_pc[provenance_bucket_index]))
#
#                 # if dataset_label == "testing" and dp == 47 and (pc == 4):  # (pc + 1 == 5 or pc + 1 == 7)#dataset_label == "testing" and
#                 #     print("\n# -- DEBUG IRINA: START -- #")
#                 #     # print("pc_scores={0}".format(pc_scores))
#                 #     print("min_pc_score_train={0}".format(min_pc_score_train))
#                 #     print("max_pc_score_train={0}".format(max_pc_score_train))
#                 #     print("cut_points_train={0}".format(cut_points_train))
#                 #     print("{0}th query point, {1}th PC, pc_score={2}".format(dp, pc, pc_score))
#                 #     print("TEST COND_1: pc_score < min_pc_score_train => {0}".format(pc_score < min_pc_score_train))
#                 #     print("int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train))) => {0}".format(int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train)))))
#                 #     print("bucket_index => {0}".format(bucket_index))
#                 #     print("provenance_bucket_index => {0}".format(provenance_bucket_index))
#                 #     print("gray_codes_pc[provenance_bucket_index] => {0}".format(gray_codes_pc[provenance_bucket_index]))
#                 #     print("TEST COND_2: pc_score > max_pc_score_train => {0}".format(pc_score > max_pc_score_train))
#                 #     print("# -- DEBUG IRINA: END -- #\n")
#
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
#             # if dataset_label == "testing" and (pc + 1 == 5 or pc + 1 == 7):
#             #     print("Points distribution to buckets per pc=" + str(pc) + " => {0}\n".format(points_distrib_per_pc))
#     # print("grey_codes_per_pc =>")
#     # print(grey_codes_per_pc)
#
#     u = np.array(data_hashcodes, dtype=bool)
#     u_compactly_binarized = compact_bit_matrix(u)
#
#     # print("u => \n", u)
#
#     return u, u_compactly_binarized
#
#
# def compress_dataset__balanced_partitioning__initial(data_train_norm, data_test_norm, sh_model, dataset_label):
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
#     # -- Create data box similarly to compress_dataset, the only upcoming difference being the partitioning with equally spaced-out line cuts instead of sine curves cuts -- #
#     data_box_train = get_sine_data_box(data_train_norm_pcaed_and_centered, sh_model, data_train_norm_n)
#     data_box_test = get_sine_data_box(data_test_norm_pcaed_and_centered, sh_model, data_test_norm_n)
#
#     # if dataset_label == "testing":
#     #     print("data_box_test => ", data_box_test[])
#
#     # print_min_max_data_box(data_box_train, "data_box_train")
#     # print_min_max_data_box(data_box_test, "data_box_test")
#
#
#     # -- Find out how many bits each pc contributes with -- #
#     bits_per_pcs = get_pc_bitwise_contribution(sh_model)
#     # print("\nbits_per_pc: {0}".format(bits_per_pcs))
#
#     if dataset_label == "training":
#         data_hashcodes = [[] for _ in range(0, data_train_norm_n)]
#     else:
#         data_hashcodes = [[] for _ in range(0, data_test_norm_n)]
#
#     grey_codes_per_pc = {}
#     for pc in range(0, sh_model.n_bits):
#         # -- Check if the Principal Component pc contributes to our hashcode generation at all -- #
#         if str(pc) in bits_per_pcs.keys():
#             # -- Establish n_buckets and n_bits -- #
#             num_bits_of_contribution = len(bits_per_pcs[str(pc)])
#             # print("{0}th PC contributes with {1} bits!".format(pc, num_bits_of_contribution))
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
#             # print("------------")
#             # print("PC_" + str(pc) + ", greycodes => ", gray_codes_pc)
#             # print_help("pc_scores", pc_scores)
#             # print_help("min_pc_score_train", min_pc_score_train)
#             # print_help("max_pc_score_train", max_pc_score_train)
#             # print_help("range_pc_train", range_pc_train)
#             # print_help("interval_pc_train", interval_pc_train)
#             # print_help("gray_codes_pc", gray_codes_pc)
#             # print_help("num_gray_codes", num_gray_codes)
#             # print("------------")
#
#             # -- Plot box/PC partitions, but ALWAYS determining cut_points for training, no matter if we compress training or testing -- #
#             cut_points_train = [min_pc_score_train + interval_pc_train * ith for ith in range(1, num_gray_codes)]
#
#
#             # -- Establish pc scores/actual scores/data box scores which is about to be partitioned -- #
#             if dataset_label == "training":
#                 # print("YES, dataset_label == training")
#                 pc_scores = data_box_train[:, pc]
#             else:
#                 # print("NO, dataset_label == testing")
#                 pc_scores = data_box_test[:, pc]
#
#             # plot_2D(np.hstack((np.vstack(pc_scores), np.zeros((len(pc_scores), 1)))), True, "Balanced", cut_points_train, "orange")
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
#
#
#                 if dataset_label == "testing" and dp == 47 and (pc == 0):  # (pc + 1 == 5 or pc + 1 == 7)#dataset_label == "testing" and
#                     print("\n# -- DEBUG IRINA: START -- #")
#                     print("pc_scores={0}".format(pc_scores))
#                     print("min_pc_score_train={0}".format(min_pc_score_train))
#                     print("max_pc_score_train={0}".format(max_pc_score_train))
#                     print("cut_points_train={0}".format(cut_points_train))
#                     print("{0}th query point, {1}th PC, pc_score={2}".format(dp, pc + 1, pc_score))
#                     print("TEST COND_1: pc_score < min_pc_score_train => {0}".format(pc_score < min_pc_score_train))
#                     print("int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train))) => {0}".format(int(math.floor((pc_score - min_pc_score_train) / (interval_pc_train)))))
#                     print("bucket_index => {0}".format(bucket_index))
#                     print("provenance_bucket_index => {0}".format(provenance_bucket_index))
#                     print("gray_codes_pc[provenance_bucket_index] => {0}".format(gray_codes_pc[provenance_bucket_index]))
#                     print("TEST COND_2: pc_score > max_pc_score_train => {0}".format(pc_score > max_pc_score_train))
#                     print("# -- DEBUG IRINA: END -- #\n")
#
#
#
#
#                 # if dp == 1 and pc == 4 and dataset_label == "testing":
#                 #     print("################################################")
#                 #     print("towards bucket_index: (pc_score - min_pc_score_train) / (interval_pc_train is {0}".format((pc_score - min_pc_score_train) / (interval_pc_train)))
#                 #     print("bucket_index uninted is {0}".format(math.floor((pc_score - min_pc_score_train) / (interval_pc_train))))
#                 #     print("2nd query point, on the 5th pc, info => ")
#                 #     print("max_pc_score_train", max_pc_score_train)
#                 #     print("min_pc_score_train", min_pc_score_train)
#                 #     print("interval_pc_train", interval_pc_train)
#                 #     print("range_pc_train", range_pc_train)
#                 #     print("provenance_bucket_index", provenance_bucket_index)
#                 #     print("gray_codes_pc[provenance_bucket_index]", gray_codes_pc[provenance_bucket_index])
#                 #     print("min_pc_scores_test", min(pc_scores))
#                 #     print("max_pc_scores_test", max(pc_scores))
#                 #     print("PC_" + str(pc) + ", cut_points_train => ", cut_points_train)
#                 #     print("pc_score", pc_score)
#                 #     print("################################################")
#
#                 data_hashcodes[dp] = np.hstack((data_hashcodes[dp], [int(bit_str) for bit_str in gray_codes_pc[provenance_bucket_index]]))
#
#                 if dataset_label == "testing" and dp == 47 and (pc == 0):
#                     print("HASHCODE after appending bits for 0th PC, data_hashcodes[dp]={0}, total_len={1}".format(data_hashcodes[dp], len(data_hashcodes[dp])))
#                 # print("pc_" + str(pc) + ", point_" + str(dp) + ", hashcode_" + str(data_hashcodes[dp]) + ", bucket-index_" + str(provenance_bucket_index) + ", bucket-index-unrounded_" + str((pc_score - min_pc_score) / (interval_pc)))
#                 points_distrib_per_pc[provenance_bucket_index] += 1
#
#             # if dataset_label == "testing" and (pc + 1 == 5 or pc + 1 == 7):
#             #     print("Points distribution to buckets per pc=" + str(pc) + " => {0}\n".format(points_distrib_per_pc))
#     # print("grey_codes_per_pc =>")
#     # print(grey_codes_per_pc)
#
#     u = np.array(data_hashcodes, dtype=bool)
#     u_compactly_binarized = compact_bit_matrix(u)
#
#     # print("u => \n", u)
#
#     return u, u_compactly_binarized
#
#
