

from src.mbn_statistics import multiple_comparison_correction
import numpy as np
import pandas as pd
import os

class Setup:
    alpha = 0.05

class Data:
    weights = os.path.join(os.getcwd(),"tests","data","weights_data1.csv")
    weights_corrected = os.path.join(os.getcwd(),"tests","data","weights_corrected_data1.csv")
    pvalue = os.path.join(os.getcwd(),"tests","data","pval_data1.csv")
    pvalue_corrected = os.path.join(os.getcwd(),"tests","data","qval_data1.csv")


def test_output_fdr():
    Setup.correction_type = "fdr_bh"
    weights = pd.read_csv(Data.weights).values
    pvalue = pd.read_csv(Data.pvalue).values

    true_weights_corrected = pd.read_csv(Data.weights_corrected)
    true_qval = pd.read_csv(Data.pvalue_corrected)

    weights_corrected,qvalue_corrected = multiple_comparison_correction(weights, pvalue, Setup)
    assert np.all([np.isclose(true_weights_corrected,weights_corrected),np.isclose(true_qval,qvalue_corrected)])


