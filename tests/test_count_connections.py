from src.ms_scheme import count_connections
import pandas as pd
import numpy as np
import os

class Data:
    weights = os.path.join(os.getcwd(),"tests","data","r_corrected.csv")
    vicinity = os.path.join(os.getcwd(),"tests","data","vicinity_graph2.npy")

def array_to_list(array):
    new_list = []
    for sub_array in array:
        new_list.append(list(sub_array[0]))
    return new_list

def test_count_connections():
    weights = pd.read_csv(Data.weights)
    true_vicinity = np.load(Data.vicinity,allow_pickle=True).tolist()
    _,est_vicinity,_ = count_connections(weights.values)
    assert true_vicinity == est_vicinity

    









