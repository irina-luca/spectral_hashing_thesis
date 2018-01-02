#!/usr/bin/env bash
# -- This runs the training phase on the 3 variants of a 3D-one-blob artificial dataset => "../../Data/Random/Small_Exp_Modes/blobs_n-100_d-3_blobs-1_seed-1" -- #

# -- 3D -- #
#filename="blobs_n-100_d-3_blobs-1_seed-1"
#filename_2="blobs_n-100_d-3_blobs-1_seed-1_squeezed_1"
#filename_3="blobs_n-100_d-3_blobs-1_seed-1_squeezed_2"
#
#python "../../train.py" -input "../../Data/Random/Small_Exp_Modes/$filename" -model "../../Results/Small_Exp_Modes/Models/$filename--bits-2" -bits 2 -log_file_train "../../Results/Small_Exp_Modes/Logs/$filename--bits-2.train.log"
#
#python "../../train.py" -input "../../Data/Random/Small_Exp_Modes/$filename_2" -model "../../Results/Small_Exp_Modes/Models/$filename_2--bits-2" -bits 2 -log_file_train "../../Results/Small_Exp_Modes/Logs/$filename_2--bits-2.train.log"
#
#python "../../train.py" -input "../../Data/Random/Small_Exp_Modes/$filename_3" -model "../../Results/Small_Exp_Modes/Models/$filename_3--bits-2" -bits 2 -log_file_train "../../Results/Small_Exp_Modes/Logs/$filename_3--bits-2.train.log"


# -- 2D -- #
filename_s="blobs_n-100_d-2_blobs-1_seed-1_squeezed"
filename_us="blobs_n-100_d-2_blobs-1_seed-1_unsqueezed"

python "../../train.py" -input "../../Data/Random/Small_Exp_Modes/$filename_s" -model "../../Results/Small_Exp_Modes/Models/$filename_s--bits-2" -bits 2 -log_file_train "../../Results/Small_Exp_Modes/Logs/$filename_s--bits-2.train.log"

python "../../train.py" -input "../../Data/Random/Small_Exp_Modes/$filename_us" -model "../../Results/Small_Exp_Modes/Models/$filename_us--bits-2" -bits 2 -log_file_train "../../Results/Small_Exp_Modes/Logs/$filename_us--bits-2.train.log"
