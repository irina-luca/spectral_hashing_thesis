import numpy as np
import pickle

if __name__ == '__main__':
    file_1_tr = "./Results/Tests__Why_is_balanced_worse_than_vanilla/Logs/d1.trarray"
    file_1_te = "./Results/Tests__Why_is_balanced_worse_than_vanilla/Logs/d1.tearray"
    file_2_tr = "./Results/Tests__Why_is_balanced_worse_than_vanilla/Logs/d2.trarray"
    file_2_tr = "./Results/Tests__Why_is_balanced_worse_than_vanilla/Logs/d2.tearray"

    arr_1 = pickle.load(open(file_1_tr, "rb"))
    arr_2 = pickle.load(open(file_2_tr, "rb"))

    for r, (r_1, r_2) in enumerate(zip(arr_1, arr_2)):
        print(r, r_1)
        print(r, r_2)
        print("")
