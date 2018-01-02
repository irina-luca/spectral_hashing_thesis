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
from random import randint

# Training && Testing usage:
# python train.py -input "./Results/Small_Exp_Buckets_Props_on_1D/Data/data_1" -model "./Results/Small_Exp_Buckets_Props_on_1D/Models/data_1_bits-8" -bits 8 -log_file_train "./Results/Small_Exp_Buckets_Props_on_1D/Logs/data_1_bits-8.train.log"

# python test.py -model "./Results/Small_Exp_Buckets_Props_on_1D/Models/data_1_bits-8" -testing "./Results/Small_Exp_Buckets_Props_on_1D/Data/data_1" -mhd 5 -k 10 -log_file_test "./Results/Small_Exp_Buckets_Props_on_1D/Logs/data_1_bits-8.test.log" -log_file_others "./Results/Small_Exp_Buckets_Props_on_1D/Logs/data_1_bits-8.others.log" -eval_type=1

def save_data_to_file(data, output_file):
    np.savetxt(output_file, data, delimiter=' ')

def main():
    # -- Experiment Info: description && motivation && data -- #
    print("# -- This is a small experiment with a dataset of bigger dim., but where only 1D contains equally distributed data on a line, while others have 0 values everywhere. -- #")
    print("# -- The plan is to run SH on this data and check props for the formed buckets, expecting that all the cuts will be made on the same PC. -- #")

    # -- Step 1: Generate data -- #
    n_samples = 1000
    n_dimensions = 4096
    random_dim_to_set = randint(0, n_dimensions)
    data_train = np.zeros((n_samples, n_dimensions))
    non_zero_dim = np.arange(n_samples)
    data_train[:, random_dim_to_set] = non_zero_dim
    # print(data_train)

    # -- Step 2: Save data_train to file -- #
    # output_file = "../../Results/Small_Exp_Buckets_Props_on_1D/Data/data_1.train"
    # save_data_to_file(data_train, output_file)

    # -- Step 3: Generate test file -- #
    data_test, color = datasets.samples_generator.make_blobs(
        n_samples=int(n_samples/10),
        n_features=n_dimensions,
        cluster_std=1,
        centers=1,
        shuffle=True,
        random_state=12)

    # -- Step 4: Save data_test to file -- #
    output_file = "../../Results/Small_Exp_Buckets_Props_on_1D/Data/data_1.test"
    save_data_to_file(data_test, output_file)

main()

