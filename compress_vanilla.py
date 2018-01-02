from helpers import *
import operator

# -- Vanilla SH Compression -- #
def compress_dataset__vanilla(data_norm, sh_model, dataset_label):
    print("\n# -- VANILLA: {0} set -- #".format(dataset_label))

    # -- Get dataset dimensions -- #
    data_norm_n, data_norm_d = data_norm.shape

    # -- PCA the given dataset according to the training set principal components -- #
    data_norm_pcaed = data_norm.dot(sh_model.pc_from_training)

    # -- Move towards the actual compression -- #
    data_norm_pcaed_and_centered = data_norm_pcaed - np.tile(sh_model.mn, (data_norm_n, 1))
    # plot_2D(data_norm_pcaed_and_centered, False, "data_norm_pcaed_and_centered")

    omegas_compress_training = sh_model.modes * np.tile(sh_model.omega_zero, (sh_model.n_bits, 1))
    u = np.zeros((data_norm_n, sh_model.n_bits), dtype=bool)

    # test_hashcode = []
    # entire_data_box = np.zeros(())
    for ith_bit in range(0, sh_model.n_bits):
        omega_i = np.tile(omegas_compress_training[ith_bit, :], (data_norm_n, 1))

        data_box = data_norm_pcaed_and_centered * omega_i + math.pi / 2
        ys = np.sin(data_norm_pcaed_and_centered * omega_i + math.pi / 2)

        # -- START: Plot sine -- #
        # if dataset_label == "testing":
        #     print_help("VANILLA, data_box, ith_bit: {0}".format(ith_bit),
        #                data_norm_pcaed_and_centered * omega_i + math.pi / 2)

        # if dataset_label == "training":
        # #     print_help("omega_i", omega_i)
        #     plot_sine_partitioning_vanilla_sh(data_norm_pcaed_and_centered, omega_i, ys)
        # -- END: Plot sine -- #

        yi = np.prod(ys, axis=1)

        # if dataset_label == "testing":
        #     nonzero_col_index = np.nonzero(omegas_compress_training[ith_bit, :])[0][0]
        #     print("data_box:\n{0}".format(data_box[:, nonzero_col_index]))
        #     print("ith_bit={0}\n{1}th PC\npc_score={2}\nbit_assigned={3}\n----------------------------------------------".format(ith_bit, nonzero_col_index, data_box[49][nonzero_col_index], (yi[49] < 0).astype(int)))  # data_box={4} / pc_scores
        #     print("")

        # u[:, ith_bit] = yi > 0
        # OBS(*): I have changed the sign here on purpose, so I get hashcodes in the same manner as the other partitioning methods (easier for comparison in debugging!)
        # if min(yi) == max(yi):
        #     print("yi={0}".format(yi))
        u[:, ith_bit] = yi < 0

    ### TRY SWAP DEBUG ###
    # if dataset_label == "testing":
    # print("DEBUG, u before swapping columns, for {0} => \n {1}".format(dataset_label, u.astype(int)))
    # u[:, [2, 1]] = u[:, [1, 2]]
    # u[:, [3, 4]] = u[:, [4, 3]]
    # print("DEBUG, u after swapping columns, for {0} => \n {1}".format(dataset_label, u.astype(int)))

    u_compactly_binarized = compact_bit_matrix(u)


    # if dataset_label == "testing":
    #     print("u[49]", u[49].astype(int))

    return u, u_compactly_binarized




