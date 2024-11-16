from typing import List, NewType, Tuple

import numpy as np
import pandas as pd

from .base_classes import Setup
from .ui_parser import TkInput as DataParser

PetData = NewType("PetData", List[Tuple[np.array, str]])


def data_import(setup: Setup) -> Tuple[PetData, str, str, str]:
    inputs = DataParser(setup)
    data = process_data(inputs.data_path, setup)

    return data, inputs.labels_path, inputs.atlas_path, inputs.coords_path


def process_data(data_path: str, setup: Setup) -> PetData:
    OPTIONS = {
        "imbalanced": process_imbalanced,
        "undersampled": process_undersampled,
        "ADASYN": process_adasyn,
    }

    process_fc = OPTIONS[setup.data_balance]

    data = process_fc(data_path)
    return data


# TODO
def process_undersampled(data_path: str) -> PetData:
    pass


# TODO
def process_adasyn(data_path: str) -> PetData:
    # imbalanced_data = process_imbalanced(data_path)
    # adasyn = ADASYN(random_state=42)
    pass


def process_imbalanced(data_path: str) -> PetData:
    data_df = pd.read_csv(data_path)
    groups = list(pd.unique(data_df.iloc[:, 0]))

    data = []
    for group in groups:
        matrix = data_df[data_df.iloc[:, 0] == group].iloc[:, 1::].values
        data.append((matrix, group))

    return data
