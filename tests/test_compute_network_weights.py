

from src.ms_scheme import gen_rows
from src.mbn_statistics import pearson_correlation, compute_network_weights
from src.data_importer import process_data
import pandas as pd
import numpy as np
import os

class Data:
    path = os.path.join(os.getcwd(),"input","dummy_mice_data.csv")
    coeff_group1 = os.path.join(os.getcwd(),"tests","data","coeff_group1.csv")
    pvalue_group1 = os.path.join(os.getcwd(),"tests","data","pvalue_group1.csv")

class Setup:
    weights = 'pearson_correlation'
    data_balance = 'imbalanced'
    random_type = 'bootstrap'


def test_coeff():
    data = process_data(Data.path, Setup)
    df = pd.read_csv(Data.coeff_group1)
    coeff, _ = pearson_correlation(data[0][0])
    assert np.all(np.isclose(df.values,coeff))

def test_ouput_values():
    data = process_data(Data.path, Setup)
    df = pd.read_csv(Data.coeff_group1)
    weights, _ = compute_network_weights(data[0][0], Setup)
    assert np.all(np.isclose(df.values,weights))

def test_correlation_order():
    np.random.seed(13)
    n_rows = 19
    data = process_data(Data.path, Setup)
    new_rows = gen_rows(Setup,n_rows)
    gen_data_sort = data[0][0][np.sort(new_rows),:]
    gen_data = data[0][0][new_rows,:]
    weights_sort, _ = compute_network_weights(gen_data_sort, Setup)
    weights, _ = compute_network_weights(gen_data, Setup)
    assert np.all(np.isclose(weights,weights_sort))


