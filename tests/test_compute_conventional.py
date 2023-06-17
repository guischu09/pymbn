
from src.mbn_builder import compute_conventional
from src.data_importer import process_data
import numpy as np
import pandas as pd
import os

class Setup:
    correction_type = "fdr_bh"
    data_balance = 'imbalanced'
    weights = 'pearson_correlation'
    threshold = 0.3


class Data:
    path = os.path.join(os.getcwd(),"input","dummy_mice_data.csv")
    conventional_r_group1 = os.path.join(os.getcwd(),"tests","data","dummy_r_conventional_group1.csv")
    conventional_r_group1_001 = os.path.join(os.getcwd(),"tests","data","dummy_r_conventional_group1_alpha0.01.csv")
    conventional_r_group2 = os.path.join(os.getcwd(),"tests","data","dummy_r_conventional_group2.csv")
    conventional_r_group2_001 = os.path.join(os.getcwd(),"tests","data","dummy_r_conventional_group2_alpha0.01.csv")


def test_conventional_r_alpha005_group1():        
    Setup.alpha = 0.05
    true_weights_group1 = pd.read_csv(Data.conventional_r_group1).values
    data = process_data(Data.path, Setup)
    weights_corrected, _ = compute_conventional(data,Setup)
    assert np.all(np.isclose(true_weights_group1,weights_corrected[:,:,0]))

def test_conventional_r_alpha001_group1():        
    Setup.alpha = 0.01
    true_weights_group1 = pd.read_csv(Data.conventional_r_group1_001).values
    data = process_data(Data.path, Setup)
    weights_corrected, _ = compute_conventional(data,Setup)
    assert np.all(np.isclose(true_weights_group1,weights_corrected[:,:,0]))

def test_conventional_r_alpha005_group2():        
    Setup.alpha = 0.05
    true_weights_group2 = pd.read_csv(Data.conventional_r_group2).values
    data = process_data(Data.path, Setup)
    weights_corrected, _ = compute_conventional(data,Setup)
    assert np.all(np.isclose(true_weights_group2,weights_corrected[:,:,1]))

def test_conventional_r_alpha001_group2():        
    Setup.alpha = 0.01
    true_weights_group2 = pd.read_csv(Data.conventional_r_group2_001).values
    data = process_data(Data.path, Setup)
    weights_corrected, _ = compute_conventional(data,Setup)
    assert np.all(np.isclose(true_weights_group2,weights_corrected[:,:,1]))
