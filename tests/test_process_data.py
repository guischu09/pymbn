import os

import numpy as np

from src.data_importer import process_data

# from src.ui_parser import ManualInput as SetupParser

# setup = SetupParser().get_parameters()


class Setup:
    pass


class Input:
    data_path1 = os.path.join(os.getcwd(), "tests", "data", "dummy_mice_data.csv")
    not_data_path = 13


def test_check_output_is_list():
    setup = Setup()
    setup.data_balance = "imbalanced"
    data = process_data(Input.data_path1, setup)
    assert type(data) == list
