
from src.ms_scheme import find_connecting_edges
import numpy as np
import os

class Data:
    vicinity2 = os.path.join(os.getcwd(),"tests","data","vicinity_graph2.npy")
    s2 = os.path.join(os.getcwd(),"tests","data","s2.npy")
    t2 = os.path.join(os.getcwd(),"tests","data","t2.npy")
    vicinity3 = os.path.join(os.getcwd(),"tests","data","vicinity_graph3.npy")
    s3 = os.path.join(os.getcwd(),"tests","data","s3.npy")
    t3 = os.path.join(os.getcwd(),"tests","data","t3.npy")

def test_find_connecting_edges1():
    true_vicinity = np.load(Data.vicinity2,allow_pickle=True).tolist()
    s2_true = np.load(Data.s2,allow_pickle=True).tolist()
    t2_true = np.load(Data.t2,allow_pickle=True).tolist()
    s_est, t_est = find_connecting_edges(true_vicinity)
    assert np.all([s_est == s2_true, t_est == t2_true])

# def test_find_connecting_edges2():
#     true_vicinity = np.load(Data.vicinity3,allow_pickle=True).tolist()
#     s3_true = np.load(Data.s3,allow_pickle=True).tolist()
#     t3_true = np.load(Data.t3,allow_pickle=True).tolist()
#     s_est, t_est = find_connecting_edges(true_vicinity)
#     assert np.all([s_est == s3_true, t_est == t3_true])
