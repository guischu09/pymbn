
from src.ui_parser import ManualSetup as SetupParser
import numpy as np

def test_compute_ms():

    setup = SetupParser().get_parameters()
    np.random.seed(setup.seed)