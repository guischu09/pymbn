from src.mbn_statistics import compute_network_weights
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

def test_coeff():
    data = process_data(Data.path, Setup)
    df = pd.read_csv(Data.coeff_group1)
    coeff, _ = compute_network_weights(data[0][0],Setup)
    assert np.all(np.isclose(df.values,coeff))

def test_frobenius_norm():
    TRUE_NORM = 9.181374446193923
    data = process_data(Data.path, Setup)
    coeff, _ = compute_network_weights(data[0][0],Setup)
    norm1 = np.linalg.norm(coeff,"fro")

    assert np.isclose(norm1,TRUE_NORM)

