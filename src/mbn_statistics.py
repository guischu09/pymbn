from typing import Tuple

import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from .base_classes import Setup


def multiple_comparison_correction(weights, pvalue, setup) -> Tuple[np.ndarray, np.ndarray]:
    # Allowed corrections are described in: https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

    _, qvalue, _, _ = multipletests(pvalue.ravel(), method=setup.correction_type)
    logic_vec = qvalue < setup.alpha
    rows, cols = np.shape(weights)
    weights_corrected = np.reshape(weights.ravel() * logic_vec.astype("uint8"), (rows, cols)) + np.eye(rows)
    qvalue_corrected = np.reshape(qvalue, (rows, cols)) - np.eye(rows)

    return weights_corrected, qvalue_corrected


def threshold_correction(
    weights_corrected: np.ndarray, pvalue_corrected: np.ndarray, setup: Setup
) -> Tuple[np.ndarray, np.ndarray]:
    logic_mat = weights_corrected < setup.threshold
    weights_corrected[logic_mat] = 0
    pvalue_corrected[logic_mat] = 1
    return weights_corrected, pvalue_corrected


def compute_network_weights(data: np.ndarray, setup: Setup) -> Tuple[np.ndarray, np.ndarray]:
    if setup.weights == "pearson_correlation":
        coeff, pvalue = pearson_correlation(data)
    else:
        raise ValueError("Invalid weight type")

    return coeff, pvalue


def pearson_correlation(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_cols = array.shape[1]
    correlations = np.zeros((n_cols, n_cols))
    pvalues = np.zeros((n_cols, n_cols))

    for i in range(n_cols):
        for j in range(i, n_cols):
            if i == j:
                correlations[i, j] = 1
                pvalues[i, j] = 1
            else:
                corr, pval = pearsonr(array[:, i], array[:, j])
                correlations[i, j] = corr
                correlations[j, i] = corr
                pvalues[i, j] = pval
                pvalues[j, i] = pval

    return correlations, pvalues
