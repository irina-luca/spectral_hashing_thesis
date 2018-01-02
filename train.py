import argparse
import numpy as np
import pickle

from helpers import *


def read_args():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-input", help="input training set", required=1)
    parser.add_argument("-model", help="model output file", required=1)
    parser.add_argument("-bits", help="bits to encode to", type=int, required=1)
    parser.add_argument("-log_file_train", help="log file for training", type=str, required=1)
    args = parser.parse_args()

    return args


def train_sh_individually(input_file, output_model, bits_to_encode_to, log_file_train_destination):
    delimiter = ' '
    data_filename_location = input_file
    data_train = np.genfromtxt(data_filename_location, delimiter=delimiter, dtype=np.float)
    print("DONE reading training set")

    data_train_norm = normalize_data(data_train)
    print("DONE normalizing training set")

    sh_model = train_sh(data_train_norm, bits_to_encode_to, data_filename_location, log_file_train_destination)
    print("DONE model creation from training set")

    pickle.dump(sh_model, open(output_model, "wb"))
    print("DONE dumping model to file: {0}".format(output_model))



def main_train():
    # -- Read args -- #
    args = read_args()

    input_file = args.input + '.train'
    output_model = args.model + '.model'
    bits_to_encode_to = args.bits
    log_file_train_destination = args.log_file_train

    train_sh_individually(input_file, output_model, bits_to_encode_to, log_file_train_destination)



main_train()

