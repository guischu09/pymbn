from src.mbn_builder import compute_representative_mbn
from src.base_classes import MSComputations
import numpy as np
import pandas as pd
import os

class Setup:
    correction_type = "fdr_bh"
    data_balance = 'imbalanced'
    weights = 'pearson_correlation'
    threshold = 0.3
    alpha = 0.05
    criteria_representation = "mean"

class Data:
    non_corrected_group1 = os.path.join(os.getcwd(),"tests","data","weights_noncorrected_mean_group1.csv")
    corrected_group1 = os.path.join(os.getcwd(),"tests","data","weights_corrected_mean_group1.csv")
    corrected_pval_group1 = os.path.join(os.getcwd(),"tests","data","pval_corrected_mean_group1.csv")
    rep_group1 = os.path.join(os.getcwd(),"tests","data","representative_mean_group1.csv")

    non_corrected_group2 = os.path.join(os.getcwd(),"tests","data","weights_noncorrected_mean_group2.csv")
    corrected_group2 = os.path.join(os.getcwd(),"tests","data","weights_corrected_mean_group2.csv")
    corrected_pval_group2 = os.path.join(os.getcwd(),"tests","data","pval_corrected_mean_group2.csv")
    rep_group2 = os.path.join(os.getcwd(),"tests","data","representative_mean_group2.csv")
    


def test_mean_group1():       

    pval_corrected = np.expand_dims(pd.read_csv(Data.corrected_pval_group1).values,2)
    pval_noncorrected = np.zeros((1,1))
    weights_corrected = np.expand_dims(pd.read_csv(Data.corrected_group1).values,2)
    weights_noncorrected = np.expand_dims(pd.read_csv(Data.non_corrected_group1).values,2)


    ms = MSComputations(weights_corrected=weights_corrected,
                        pval_corrected=pval_corrected,
                        weights_noncorrected=weights_noncorrected,
                        pval_noncorrected=pval_noncorrected)

    
    
    true_mean_group1 = pd.read_csv(Data.rep_group1).values

    weights_corrected, _ = compute_representative_mbn(ms,Setup)
    assert np.all(np.isclose(true_mean_group1,weights_corrected[:,:,0]))


def test_mean_group2():       

    pval_corrected = np.expand_dims(pd.read_csv(Data.corrected_pval_group2).values,2)
    pval_noncorrected = np.zeros((1,1))
    weights_corrected = np.expand_dims(pd.read_csv(Data.corrected_group2).values,2)
    weights_noncorrected = np.expand_dims(pd.read_csv(Data.non_corrected_group2).values,2)


    ms = MSComputations(weights_corrected=weights_corrected,
                        pval_corrected=pval_corrected,
                        weights_noncorrected=weights_noncorrected,
                        pval_noncorrected=pval_noncorrected)

    
    
    true_mean_group2 = pd.read_csv(Data.rep_group2).values

    weights_corrected, _ = compute_representative_mbn(ms,Setup)
    assert np.all(np.isclose(true_mean_group2,weights_corrected[:,:,0]))