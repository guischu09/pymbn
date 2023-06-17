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
