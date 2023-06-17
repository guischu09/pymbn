
from src.ms_scheme import find_bins_degree_distribution
import numpy as np
import pandas as pd
import os

class Data:
    edge1 = os.path.join(os.getcwd(),"tests","data","edge_combination1.csv")
    edge2 = os.path.join(os.getcwd(),"tests","data","edge_combination2.csv")


def test_check_outputs():
    n_nonrep, n_nodes = 91, 14
    e1 = pd.read_csv(Data.edge1)
    e2 = pd.read_csv(Data.edge2)
    edge1, edge2 = find_bins_degree_distribution(n_nonrep, n_nodes)
    assert np.all(e1.values == edge1) and np.all(e2.values == edge2)




