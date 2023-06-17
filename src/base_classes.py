from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class InputHelper(ABC):
    @abstractmethod
    def get_data_filepath(self) -> str:
        pass

    @abstractmethod
    def get_coords_filepath(self) -> str:
        pass

    @abstractmethod
    def get_atlas_filepath(self) -> str:
        pass

    @abstractmethod
    def get_labels_filepath(self) -> str:
        pass


@dataclass
class Setup:
    alpha: float
    theta: float
    threshold: float
    weights: str
    correction_type: str
    criteria_representation: str
    data_balance: str
    n_samples_measures: int
    plot_3d: bool
    brain_type: str
    which_plot: str
    output_format: str
    mbn_method: str
    probability_treshold: float
    n_samples: int
    random_type: str
    interactive: bool
    seed: int


class SetupHelper(ABC):
    @abstractmethod
    def get_parameters(self) -> Setup:
        pass


@dataclass
class MSComputations:
    weights_corrected: np.ndarray
    pval_corrected: np.ndarray
    weights_noncorrected: np.ndarray
    pval_noncorrected: np.ndarray
