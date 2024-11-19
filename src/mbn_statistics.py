from typing import Tuple

import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from scipy import stats

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
    """Vectorized implementation using scipy and matrix operations"""
    n_rows, n_cols = array.shape

    # Standardize the data
    standardized = (array - array.mean(axis=0)) / array.std(axis=0, ddof=1)

    # Compute correlation matrix
    correlations = (standardized.T @ standardized) / (n_rows - 1)

    # Compute t-statistic
    t_stat = correlations * np.sqrt((n_rows - 2) / (1 - correlations**2))

    # Compute p-values
    pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_rows - 2))

    # Fix diagonal values
    np.fill_diagonal(correlations, 1.0)
    np.fill_diagonal(pvalues, 1.0)

    return correlations, pvalues
