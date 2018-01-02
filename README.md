# -- Spectral Hashing: Thesis -- #
# Training: Usage example =>
python train.py -input "./Data/Handmade/h1" -model "./Results/Handmade/Models/h1__1_bits-2" -bits 2 -log_file_train "./Results/Handmade/Logs/h1__1_bits-2.train.log"

# Testing: Usage example =>
python test.py -model "./Results/Handmade/Models/h1__1_bits-2" -testing "./Data/Handmade/h1" -mhd 5 -k 10 -log_file_test "./Results/Handmade/Logs/h1__1_bits-2.test.balanced.uord.log" -log_file_others "./Results/Handmade/Logs/h1__1_bits-2.others.balanced.log" -eval_type=1 -compress_type="balanced" -ordered_pcs="uord"
