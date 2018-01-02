#!/usr/bin/env bash

data_folders=("MNIST" "Peekaboom" "SIFT" "LabelMe")
data_filenames=("mnist.data" "PeekaboomGist.data" "ANN_SIFT1M_128D.data" "LabelMeGist_total.data")
dataset_sizes=(70000 57637 1000000 22019)

#sample_sizes_common=( 1000 5000 10000 20000 30000 40000 50000 )
#sample_sizes_exception=( 1000 5000 10000 15000 20000 )

for i in $(seq 0 3);
do
    echo ${data_folders[$i]}
    if [ ! -d ./Data/${data_folders[$i]}/Samples ]; then
      mkdir -p ./Data/${data_folders[$i]}/Samples;
    fi
    sample_sizes=( 1000 5000 10000 20000 30000 40000 50000 )
    if [ ${data_folders[$i]} = "LabelMe" ]; then
       sample_sizes=( 1000 5000 10000 15000 20000 )
    fi
    for ss in ${sample_sizes[@]};
    do
        ts=$((ss / 10))
        python sample-data.py -i "./Data/${data_folders[$i]}/${data_filenames[$i]}" -dataset_size ${dataset_sizes[$i]} -ss_or_fr "ss" -ss ${ss} -num_s=1 -ext ".train"
        python sample-data.py -i "./Data/${data_folders[$i]}/${data_filenames[$i]}" -dataset_size ${dataset_sizes[$i]} -ss_or_fr "ss" -ss ${ts} -num_s=1 -ext ".test"
    done
done