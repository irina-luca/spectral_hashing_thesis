#!/usr/bin/env bash

sample_sizes=( 1000 )
eval_types=( 0 1 )
sample_size_testing_fraction=10
num_testing_samples=1
for sample_size in "${sample_sizes[@]}"
do
    # -- Testing: for each bits_to_encode_to, we create 1 model -- #
    loop_bits=( 8 16 32 64 128 256 )
    input_model='./Results/Small_Exp_Outliers/Models/blob-outlier_ss=1000_d=4096__'


    for bits in "${loop_bits[@]}"
    do
        for eval_type in "${eval_types[@]}"
        do
            echo "TESTING for bits_to_encode_to = ${bits}"
            echo "we read model "${input_model}_bits-${bits}
            echo ${input_model}_bits-${bits}".model"

            python test.py -model "${input_model}_bits-${bits}" -testing "./Results/Small_Exp_Outliers/Data/blob-outlier_ss=100_d=4096" -mhd 30 -k 100 -log_file_test "./Results/Small_Exp_Outliers/Logs/blob-outlier_ss=1000_d=4096__bits-${bits}.test.log" -log_file_others "./Results/Small_Exp_Outliers/Logs/blob-outlier_ss=1000_d=4096___bits-${bits}.others.log" -eval_type=1
            python test.py -model "${input_model}_bits-${bits}" -testing "./Results/Small_Exp_Outliers/Data/blob-outlier_ss=100_d=4096" -mhd 30 -k 100 -log_file_test "./Results/Small_Exp_Outliers/Logs/blob-outlier_ss=1000_d=4096__bits-${bits}.test.log" -log_file_others "./Results/Small_Exp_Outliers/Logs/blob-outlier_ss=1000_d=4096___bits-${bits}.others.log" -eval_type=0
        done
    done # done loop_bits loop for training
done # done sample_sizes loop
