# -- For h2.train/.test -- #
python train.py -input "./Data/Handmade/h2" -model "./Results/Handmade/Models/h2_bits-2" -bits 2 -log_file_train "./Results/Handmade/Logs/h2_bits-2.train.log"

python test.py -model "./Results/Handmade/Models/h2_bits-2" -testing "./Data/Handmade/h2" -mhd 5 -k 3 -log_file_test "./Results/Handmade/Logs/h2_bits-2.test.log" -log_file_others "./Results/Handmade/Logs/h2_bits-2.others.log" -eval_type=1



# -- For h1.train/.test -- #
python train.py -input "./Data/Handmade/h1" -model "./Results/Handmade/Models/h1_bits-2" -bits 2 -log_file_train "./Results/Handmade/Logs/h1_bits-2.train.log"

python test.py -model "./Results/Handmade/Models/h1_bits-2" -testing "./Data/Handmade/h1" -mhd 5 -k 3 -log_file_test "./Results/Handmade/Logs/h1_bits-2.test.log" -log_file_others "./Results/Handmade/Logs/h1_bits-2.others.log" -eval_type=1

