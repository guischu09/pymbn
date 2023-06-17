from src.ms_scheme import gen_rows
import numpy as np
import os


class Setup:
    random_type = 'bootstrap'

class Data:
    path = os.path.join(os.getcwd(),"tests","data","temp_array.npy")

def test_reproducibility_rand():
    np.random.seed(13)
    n_rows = 77
    true_rows = np.load(Data.path,allow_pickle=True)
    new_rows1 = gen_rows(Setup,n_rows)
    
    assert np.all(np.isclose(new_rows1,true_rows))