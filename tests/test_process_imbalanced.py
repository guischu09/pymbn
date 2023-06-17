import os

import numpy as np
import pytest

from src.data_importer import PetData, process_imbalanced


class Input:
    data_path1 = os.path.join(os.getcwd(), "tests", "data", "dummy_mice_data.csv")
    not_data_path = 13


def test_check_output_type_list():
    data = process_imbalanced(Input.data_path1)
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
