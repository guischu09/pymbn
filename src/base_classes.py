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
    mbn_method: str
    alpha: float
    theta: float
    threshold: float
    weights: str
    correction_type: str
    sampling_type: str
    n_samples: int
    criteria_representation: str
    data_balance: str
    n_samples_measures: int
    plot_3d: bool
    interactive: bool
    brain_type: str
    plot_heatmap: bool
    plot_circle: bool
    output_format: str
    probability_treshold: float
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
