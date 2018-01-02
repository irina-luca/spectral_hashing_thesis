
#!/usr/bin/env bash

# Usage example:  sh ./Bash_Scripts/evaluate-mnist.sh


# -- 0.0 Generalities/ Observations -- #
data_folder="Profi"
data_filename="profiset.tabbed.txt"
dataset_size=20000001

# Obs. (1) => in training, change -input name depending on ss or fr

# -- 0.1 Make folders that don't exist -- #
if [ ! -d ./Data/${data_folder}/Samples ]; then
  mkdir -p ./Data/${data_folder}/Samples;
fi
if [ ! -d ./Results/${data_folder} ]; then
  mkdir -p ./Results/${data_folder};
fi
if [ ! -d ./Results/${data_folder}/Logs ]; then
  mkdir -p ./Results/${data_folder}/Logs;
fi
if [ ! -d ./Results/${data_folder}/Models ]; then
  mkdir -p ./Results/${data_folder}/Models;
fi

# -- 1. Sampling data -- #
sample_sizes=( 1000 5000 10000 20000 30000 40000 50000 100000)
sample_size_testing_fraction=10
num_testing_samples=1
for sample_size in "${sample_sizes[@]}"
do
    sample_size_testing=$(($sample_size/$sample_size_testing_fraction))
    #echo $sample_size_testing
    #echo $sample_size
    python sample-data.py -i "./Data/${data_folder}/${data_filename}" -dataset_size $dataset_size -ss_or_fr "ss" -ss $sample_size -ext ".train" -seed 12 -num_s 1 &
    python sample-data.py -i "./Data/${data_folder}/${data_filename}" -dataset_size $dataset_size -ss_or_fr "ss" -ss $sample_size_testing -ext ".test" -seed 22 -num_s $num_testing_samples &
    echo "DONE sampling training && testing for ${sample_size}"
done # done sample_sizes loop
