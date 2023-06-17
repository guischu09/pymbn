import os

import numpy as np


class Data:
    vicinity2 = os.path.join(os.getcwd(), "tests", "data", "vicinity_graph2.npy")
    s2 = os.path.join(os.getcwd(), "tests", "data", "s2.npy")
    t2 = os.path.join(os.getcwd(), "tests", "data", "t2.npy")
    vicinity3 = os.path.join(os.getcwd(), "tests", "data", "vicinity_graph3.npy")
    s3 = os.path.join(os.getcwd(), "tests", "data", "s3.npy")
    t3 = os.path.join(os.getcwd(), "tests", "data", "t3.npy")


# def find_connecting_edges(vicinity_graph):

#     n_nodes = len(vicinity_graph)
#     edges = []
#     for i in range(n_nodes):
#         for j in range(len(vicinity_graph[i])):
#             edges.append ([i, vicinity_graph[i][j]])

#     if 'edges' in locals():
#         edges = np.sort(edges, axis=1)
#         _, idx, counts = np.unique(edges, axis=0, return_counts=True, return_index=True)
#         idx_sorted = np.argsort(idx)
#         edges = edges[idx[idx_sorted[:len(idx_sorted)]]]
#         node_s = edges[:,0]; node_t = edges[:,1]
#     else:
#         edges = np.array([[0, 0]])
#         node_s = edges[:,0]; node_t = edges[:,1]

#     return node_s, node_t


def find_connecting_edges(vicinity_graph):
    n_nodes = len(vicinity_graph)
    edges = None

    for i in range(n_nodes):
        edges_i = [[i, j] for j in vicinity_graph[i]]
        edges = edges_i if edges is None else np.concatenate((edges, edges_i))

    edges = np.sort(edges, axis=1)
    _, idx, counts = np.unique(edges, axis=0, return_counts=True, return_index=True)
    edges = edges[np.sort(idx)]

    node_s = edges[:, 0]
    node_t = edges[:, 1]

    return node_s, node_t


def test_find_connecting_edges():
    true_vicinity = np.load(Data.vicinity, allow_pickle=True).tolist()
    s_true = np.squeeze(np.load(Data.s, allow_pickle=True))
    t_true = np.squeeze(np.load(Data.t, allow_pickle=True))
    s_est, t_est = find_connecting_edges(true_vicinity)
    assert np.all([np.isclose(s_est, s_true), np.isclose(t_est, t_true)])


def test_find_connecting_edges2():
    true_vicinity = np.load(Data.vicinity3, allow_pickle=True).tolist()
    s3_true = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        8,
        8,
        10,
        12,
    ]
    t3_true = [
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        3,
        12,
        13,
        5,
        9,
        12,
        13,
        5,
        6,
        7,
        8,
        9,
        10,
        6,
        7,
        8,
        9,
        7,
        8,
        9,
        10,
        11,
        8,
        9,
        10,
        11,
        9,
        10,
        11,
        13,
    ]
    s_est, t_est = find_connecting_edges(true_vicinity)
    assert np.all([list(s_est) == s3_true, list(t_est) == t3_true])


import numpy as np

from src.ms_scheme import gen_rows


class Setup:
    random_type = "bootstrap"


class Data_rep:
    rep = os.path.join(os.getcwd(), "tests", "data", "temp_array.npy")


def test_reproducibility_rand():
    np.random.seed(13)
    n_rows = 77
    if not os.path.exists(Data_rep.rep):
        new_rows1 = gen_rows(Setup, n_rows)
        np.save(Data_rep.rep, new_rows1)


##############################################################

import os

import numpy as np
import pandas as pd

from src.mbn_statistics import multiple_comparison_correction


class Setup:
    alpha = 0.05


class Data:
    weights = os.path.join(os.getcwd(), "tests", "data", "weights_data1.csv")
    weights_corrected = os.path.join(os.getcwd(), "tests", "data", "weights_corrected_data1.csv")
    pvalue = os.path.join(os.getcwd(), "tests", "data", "pval_data1.csv")
    pvalue_corrected = os.path.join(os.getcwd(), "tests", "data", "qval_data1.csv")

    weights_corrected_bonf = os.path.join(os.getcwd(), "tests", "data", "weights_corrected_bonf_data1.csv")
    pvalue_corrected_bonf = os.path.join(os.getcwd(), "tests", "data", "qval_bonf_data1.csv")


def test_output_fdr():
    Setup.correction_type = "fdr_bh"
    weights = pd.read_csv(Data.weights).values
    pvalue = pd.read_csv(Data.pvalue).values

    true_weights_corrected = pd.read_csv(Data.weights_corrected)
    true_qval = pd.read_csv(Data.pvalue_corrected)

    weights_corrected, qvalue_corrected = multiple_comparison_correction(weights, pvalue, Setup)
    assert np.all([np.isclose(true_weights_corrected, weights_corrected), np.isclose(true_qval, qvalue_corrected)])


def test_output_bonferroni():
    Setup.correction_type = "bonferroni"
    weights = pd.read_csv(Data.weights).values
    pvalue = pd.read_csv(Data.pvalue).values

    true_weights_corrected = pd.read_csv(Data.weights_corrected_bonf)
    true_qval = pd.read_csv(Data.pvalue_corrected_bonf)

    weights_corrected, qvalue_corrected = multiple_comparison_correction(weights, pvalue, Setup)
    assert np.all([np.isclose(true_weights_corrected, weights_corrected), np.isclose(true_qval, qvalue_corrected)])


###########################################################################################################

import os

import numpy as np
import pandas as pd

from src.data_importer import process_data
from src.mbn_statistics import compute_network_weights, pearson_correlation
from src.ms_scheme import gen_rows
from src.ui_parser import ManualSetup as SetupParser


class Data:
    path = os.path.join(os.getcwd(), "input", "dummy_mice_data.csv")
    coeff_group1 = os.path.join(os.getcwd(), "tests", "data", "coeff_group1.csv")
    pvalue_group1 = os.path.join(os.getcwd(), "tests", "data", "pvalue_group1.csv")


class Setup:
    weights = "pearson_correlation"
    balanced = "imbalanced"
    random_type = "bootstrap"


def test_coeff():
    data = process_data(Data.path, Setup)
    df = pd.read_csv(Data.coeff_group1)
    coeff, _ = pearson_correlation(data[0][0])
    assert np.all(np.isclose(df.values, coeff))


def test_ouput_values():
    data = process_data(Data.path, Setup)
    df = pd.read_csv(Data.coeff_group1)
    weights, _ = compute_network_weights(data[0][0], Setup)
    assert np.all(np.isclose(df.values, weights))


def test_correlation_order():
    np.random.seed(13)
    n_rows = 19
    data = process_data(Data.path, Setup)
    new_rows = gen_rows(Setup, n_rows)
    gen_data_sort = data[0][0][np.sort(new_rows), :]
    gen_data = data[0][0][new_rows, :]
    weights_sort, _ = compute_network_weights(gen_data_sort, Setup)
    weights, _ = compute_network_weights(gen_data, Setup)


#############################################################################################################


import os

import numpy as np
import pandas as pd

from src.data_importer import process_data
from src.mbn_builder import compute_conventional


class Setup:
    correction_type = "fdr_bh"
    data_balance = "imbalanced"
    weights = "pearson_correlation"
    threshold = 0.3


class Data:
    path = os.path.join(os.getcwd(), "input", "dummy_mice_data.csv")
    conventional_r_group1 = os.path.join(os.getcwd(), "tests", "data", "dummy_r_conventional_group1.csv")
    conventional_r_group2 = os.path.join(os.getcwd(), "tests", "data", "dummy_r_conventional_group2.csv")


def test_conventional_r_alpha005_group1():
    Setup.alpha = 0.05
    true_weights_group1 = pd.read_csv(Data.conventional_r_group1).values
    data = process_data(Data.path, Setup)
    weights_corrected, _ = compute_conventional(data, Setup)
    assert np.all(np.isclose(true_weights_group1, weights_corrected[:, :, 0]))


####################################################################################


import os

import numpy as np
import pandas as pd

from src.base_classes import MSComputations
from src.data_importer import process_data
from src.mbn_builder import compute_representative_mbn


class Setup:
    correction_type = "fdr_bh"
    data_balance = "imbalanced"
    weights = "pearson_correlation"
    threshold = 0.3
    alpha = 0.05
    criteria_representation = "mean"


class Data:
    non_corrected_group1 = os.path.join(os.getcwd(), "tests", "data", "weights_noncorrected_mean_group1.csv")
    corrected_group1 = os.path.join(os.getcwd(), "tests", "data", "weights_corrected_mean_group1.csv")
    corrected_pval_group1 = os.path.join(os.getcwd(), "tests", "data", "pval_corrected_mean_group1.csv")
    rep_group1 = os.path.join(os.getcwd(), "tests", "data", "representative_mean_group1.csv")

    non_corrected_group2 = os.path.join(os.getcwd(), "tests", "data", "weights_noncorrected_mean_group2.csv")
    corrected_group2 = os.path.join(os.getcwd(), "tests", "data", "weights_corrected_mean_group2.csv")
    corrected_pval_group2 = os.path.join(os.getcwd(), "tests", "data", "pval_corrected_mean_group2.csv")
    rep_group2 = os.path.join(os.getcwd(), "tests", "data", "representative_mean_group2.csv")


def test_mean_group1():
    pval_corrected = np.expand_dims(pd.read_csv(Data.corrected_pval_group1).values, 2)
    pval_noncorrected = np.zeros((1, 1))
    weights_corrected = np.expand_dims(pd.read_csv(Data.corrected_group1).values, 2)
    weights_noncorrected = np.expand_dims(pd.read_csv(Data.non_corrected_group1).values, 2)

    ms = MSComputations(
        weights_corrected=weights_corrected,
        pval_corrected=pval_corrected,
        weights_noncorrected=weights_noncorrected,
        pval_noncorrected=pval_noncorrected,
    )

    true_mean_group1 = pd.read_csv(Data.rep_group1).values

    weights_corrected, _ = compute_representative_mbn(ms, Setup)
    assert np.all(np.isclose(true_mean_group1, weights_corrected[:, :, 0]))


##################################################################################################
import os
import pickle

import pytest

from src.data_importer import PetData, process_imbalanced


class Input2:
    data_path = os.path.join(os.getcwd(), "tests", "data", "dummy_mice_data.csv")
    not_data_path = 13


def test_check_output_type_list():
    data = process_imbalanced(Input2.data_path)
    assert type(data) == list


def test_check_type_of_output_list_tuple():
    data = process_imbalanced(Input.data_path1)
    for d in data:
        assert type(d) == tuple


def test_check_type_of_output_list_tuple_index0():
    data = process_imbalanced(Input.data_path1)
    for d in data:
        assert type(d[0]) == np.ndarray


def test_check_type_of_output_list_tuple_index1():
    data = process_imbalanced(Input.data_path1)
    for d in data:
        assert type(d[1]) == str


#############################################################################################3
import os
import pickle

import numpy as np

from src.ms_scheme import sample_data
from src.ui_parser import ManualSetup as SetupParser


class Input:
    data_path = os.path.join(os.getcwd(), "tests", "data", "pet_data_demo.pickle")
    sample_output_path = os.path.join(os.getcwd(), "tests", "data", "sample_data_demo.pickle")


def read_input_pickle():
    with open(Input.data_path, "rb") as f:
        return pickle.load(f)


def test_output_type():
    setup = SetupParser().get_parameters()
    with open(Input.data_path, "rb") as f:
        data = pickle.load(f)
    data_group = data[0]
    output = sample_data(data_group, setup)
    assert type(output) == tuple


def test_output_types_are_arrays():
    setup = SetupParser().get_parameters()
    with open(Input.data_path, "rb") as f:
        data = pickle.load(f)
    data_group = data[0]
    output = sample_data(data_group, setup)
    assert np.all(
        [
            type(output[0]) == np.ndarray,
            type(output[1]) == np.ndarray,
            type(output[2]) == np.ndarray,
            type(output[3]) == np.ndarray,
        ]
    )


def test_array_values():
    setup = SetupParser().get_parameters()
    np.random.seed(setup.seed)
    with open(Input.data_path, "rb") as f:
        data = pickle.load(f)
    with open(Input.sample_output_path, "rb") as f:
        true_output = pickle.load(f)
    data_group = data[0]
    output = sample_data(data_group, setup)

    for out1, out2 in zip(output, true_output):
        assert np.all(np.isclose(out1.ravel(), out2.ravel()))


if __name__ == "__main__":
    test_array_values()
