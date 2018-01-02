#!/usr/bin/env bash

generate_data=true
target_g="Dataset_Generators/generators__random/blobs.py"
target_folder_tt="Tests__Why_is_balanced_worse_than_vanilla"
generated_data_filename="randomblob_8"
n_test=100
n_train=100
d=8
seed_1=14
seed_2=14

bits=2
k=10
mhd=5
eval_type=1
method="vb"

methods_to_compare="vanilla_balanced"

if [ "$generate_data" == true ] ; then
    python ${target_g} -n ${n_train} -d ${d} -seed ${seed_1} -blobs 1 -output "Data/${target_folder_tt}/${generated_data_filename}.train" -if_color_col 0
    python ${target_g} -n ${n_test} -d ${d} -seed ${seed_2} -blobs 1 -output "Data/${target_folder_tt}/${generated_data_filename}.test" -if_color_col 0
    echo "Generated testing set."

#    python ${target_g} -n ${n_train} -d ${d} -seed ${seed} -blobs 1 -output "Data/${target_folder_tt}/${generated_data_filename}.test" -if_color_col 0
#    echo "Generated training set."
fi


if [ ! -d ./Results/${target_folder_tt}/GTs ]; then
  mkdir -p ./Results/${target_folder_tt}/GTs;
fi
if [ ! -d ./Results/${target_folder_tt}/Logs ]; then
  mkdir -p ./Results/${target_folder_tt}/Logs;
fi
if [ ! -d ./Results/${target_folder_tt}/Models ]; then
  mkdir -p ./Results/${target_folder_tt}/Models;
fi

rm ./Results/${target_folder_tt}/Logs/*

python train.py -input "./Data/${target_folder_tt}/${generated_data_filename}" -model "./Results/${target_folder_tt}/Models/${generated_data_filename}_bits-${bits}" -bits ${bits} -log_file_train "./Results/${target_folder_tt}/Logs/${generated_data_filename}_bits-${bits}.train.log"
echo "Trained."


if [[ $method == *"v"* ]]; then
    python test.py -model "./Results/${target_folder_tt}/Models/${generated_data_filename}_bits-${bits}" -testing "./Data/${target_folder_tt}/${generated_data_filename}" -mhd ${mhd} -k ${k} -log_file_test "./Results/${target_folder_tt}/Logs/${generated_data_filename}_bits-${bits}.test.vanilla.log" -log_file_others "./Results/${target_folder_tt}/Logs/${generated_data_filename}_bits-${bits}.others.vanilla.log" -eval_type=${eval_type} -compress_type="vanilla" > "./Results/${target_folder_tt}/Logs/${generated_data_filename}.vanilla.bits-${bits}.debug.log"
    echo "Tested with vanilla."
fi

if [[ $method == *"b"* ]]; then
    python test.py -model "./Results/${target_folder_tt}/Models/${generated_data_filename}_bits-${bits}" -testing "./Data/${target_folder_tt}/${generated_data_filename}" -mhd ${mhd} -k ${k} -log_file_test "./Results/${target_folder_tt}/Logs/${generated_data_filename}_bits-${bits}.test.balanced.log" -log_file_others "./Results/${target_folder_tt}/Logs/${generated_data_filename}_bits-${bits}.others.balanced.log" -eval_type=${eval_type} -compress_type="balanced" > "./Results/${target_folder_tt}/Logs/${generated_data_filename}.balanced.bits-${bits}.debug.log"
  echo "Tested with balanced."
fi

if [[ $method == *"m"* ]]; then
    python test.py -model "./Results/${target_folder_tt}/Models/${generated_data_filename}_bits-${bits}" -testing "./Data/${target_folder_tt}/${generated_data_filename}" -mhd ${mhd} -k ${k} -log_file_test "./Results/${target_folder_tt}/Logs/${generated_data_filename}_bits-${bits}.test.median.log" -log_file_others "./Results/${target_folder_tt}/Logs/${generated_data_filename}_bits-${bits}.others.median.log" -eval_type=${eval_type} -compress_type="median" > "./Results/${target_folder_tt}/Logs/${generated_data_filename}.median.bits-${bits}.debug.log"
    echo "Tested with median."
fi


# -- This is not quite general!!!! -- #
echo "Start: Check buckets' indices difference"
python debug.py -input "./Results/${target_folder_tt}/Logs/${generated_data_filename}" -compress_types ${methods_to_compare} > "./Results/${target_folder_tt}/Logs/${generated_data_filename}.compare.${methods_to_compare}.debug.log"
echo "End: Check buckets' indices difference"
