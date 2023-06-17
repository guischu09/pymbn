from typing import Tuple

import numpy as np

from .base_classes import MSComputations, Setup
from .conventional_method import compute_conventional
from .data_importer import PetData
from .ms_scheme import compute_ms


def build_network(data: PetData, setup: Setup) -> Tuple[np.ndarray, np.ndarray]:
    ms_data = compute_ms(data, setup)
    if setup.mbn_method == "ms_scheme":
        group_networks, _ = compute_representative_mbn(ms_data, setup)
    else:
        group_networks, _ = compute_conventional(data, setup)

    return group_networks, ms_data.weights_corrected


def compute_representative_mbn(ms_data: MSComputations, setup: Setup):
    OPTIONS = {"mean": mean_mbn, "mode": mode_mbn, "geodesic": geodesic_mbn}

    process_fc = OPTIONS[setup.criteria_representation]

    data = process_fc(ms_data)
    return data


def mean_mbn(ms: MSComputations):
    # data info:
    n_row, _, n_class = np.shape(ms.weights_noncorrected)

    # Allocate memory
    real_mean_corr = np.zeros((n_row, n_class))
    corr_index = np.zeros(n_class, dtype=int)
    dim = int(np.sqrt(n_row))
    weights_representative = np.zeros((dim, dim, n_class))
    pval_representative = np.zeros((dim, dim, n_class))

    for u in range(n_class):
        real_mean_corr[:, u] = np.nanmean(ms.weights_noncorrected[:, :, u], axis=1)

        diff = ms.weights_noncorrected[:, :, u] - np.expand_dims(real_mean_corr[:, u], axis=1)
        dist = np.sqrt(np.sum(diff**2, axis=0))
        corr_index[u] = np.argmin(dist)

        r_vector = ms.weights_corrected[:, corr_index[u], u]
        pval_vector = ms.pval_corrected[:, corr_index[u], u]

        weights_representative[:, :, u] = np.reshape(r_vector, (dim, dim))
        pval_representative[:, :, u] = np.reshape(pval_vector, (dim, dim))

    return weights_representative, pval_representative


# TODO
def mode_mbn(ms_data: MSComputations):
    pass


# TODO
def geodesic_mbn(ms: MSComputations):
    # data info:
    n_row, n_samples, n_class = np.shape(ms.weights_noncorrected)
    dim = int(np.sqrt(n_row))

    # Allocate memory
    real_mean_mats = np.zeros((dim, dim, n_class))
    weights_representative = np.zeros((dim, dim, n_class))
    pval_representative = np.zeros((dim, dim, n_class))

    for u in range(n_class):
        # Compute real mean correlation matrix
        real_mean_corr = np.nanmean(ms.weights_noncorrected[:, :, u], axis=1)
        real_mean_mats[:, :, u] = real_mean_corr.reshape(dim, dim)

    mats_list = real_mean_mats.transpose((2, 0, 1))

    # Compute geodesic distances
    geo_dists = np.zeros((n_class, n_samples))
    for u in range(n_class):
        for vv in range(n_samples):
            Q1 = ms.weights_noncorrected[:, vv, u].reshape(dim, dim)

            # Regularize matrices equally
            reg_mats_cell, _ = regularize_matrices([mats_list[u], Q1])
            geo_dists[u, vv] = f_dist_geodesic(reg_mats_cell[0], reg_mats_cell[1])

    min_dist = np.min(geo_dists, axis=1)

    for u in range(n_class):
        corr_index = np.where(geo_dists[u] == min_dist[u])[0][0]

        weights_vector = ms.weights_corrected[:, corr_index, u]
        pval_vector = ms.pval_corrected[:, corr_index, u]

        weights_representative[:, :, u] = weights_vector.reshape(dim, dim)
        pval_representative[:, :, u] = pval_vector.reshape(dim, dim)

    return weights_representative, pval_representative


def regularize_matrices(mats_list):
    tau = 0
    n_mats = len(mats_list)
    flag = np.zeros(n_mats, dtype=bool)
    dim = mats_list[0].shape[0]

    for u in range(n_mats):
        _, flag[u] = np.linalg.eigh(mats_list[u])

    while any(flag):
        for v in range(n_mats):
            mats_list[v] = mats_list[v] + tau * np.eye(dim)
            _, flag[v] = np.linalg.eigh(mats_list[v])

        if any(flag):
            tau += 0.1

    reg_mats_cell = [mats_list[u] for u in range(n_mats)]

    return reg_mats_cell, tau


def f_dist_geodesic(Q1, Q2):
    # This function can be used to compute Geodesic Distance between
    # two Symmetric Positive Definite (SPD) FCs.

    Q = np.linalg.solve(Q1, Q2)
    e = np.linalg.eigvalsh(Q)
    dg = np.sqrt(np.sum(np.log(e) ** 2))

    return dg
