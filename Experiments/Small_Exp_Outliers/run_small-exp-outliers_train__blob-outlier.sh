#!/usr/bin/env bash

data_source="./Results/Small_Exp_Outliers/Data/blob-outlier_ss=1000_d=4096"

sample_sizes=( 1000 )
sample_size_testing_fraction=10
num_testing_samples=1
for sample_size in "${sample_sizes[@]}"
do
    # -- 2. Training: for each bits_to_encode_to, we create 1 model -- #
    loop_bits=( 8 16 32 64 128 256 )
    input_model='./Results/Small_Exp_Outliers/Models/blob-outlier_ss=1000_d=4096__'
    input_log='./Results/Small_Exp_Outliers/Logs/blob-outlier_ss=1000_d=4096__'

    for bits in "${loop_bits[@]}"
    do
        echo "TRAINING for bits_to_encode_to = ${bits}"
        python train.py -input "${data_source}" -model "${input_model}_bits-${bits}" -bits ${bits} -log_file_train "${input_log}_bits-${bits}.train.log"
    done # done loop_bits loop for training
done # done sample_sizes loop
