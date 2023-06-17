import os
from typing import List

import bct
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base_classes import Setup
from .data_importer import PetData

MEASURES = [
    "global_efficiency_bin",
    "global_efficiency_wei",
    "assortativity_coeff_bin",
    "assortativity_coeff_wei",
    "average_degree",
    "average_strength",
    "density",
    "clustering_coeff_bin",
    "clustering_coeff_wei",
    "small_worldness_sigma",
]
# "small_worldness_omega"]


def global_efficiency_bin(network) -> float:
    return bct.efficiency_bin(network, local=False)


def global_efficiency_wei(network) -> float:
    return bct.efficiency_wei(network, local=False)


def assortativity_coeff_bin(network) -> float:
    return bct.assortativity_bin(network, flag=0)


def assortativity_coeff_wei(network) -> float:
    return bct.assortativity_wei(network, flag=0)


def average_degree(network) -> float:
    return np.mean(bct.degrees_und(network))


def average_strength(network) -> float:
    return np.mean(bct.strengths_und(network))


def density(network) -> float:
    return bct.density_und(network)


def clustering_coeff_bin(network) -> float:
    return bct.clustering_coef_bu(network)


def clustering_coeff_wei(network) -> float:
    return bct.clustering_coef_wu(network)


def small_worldness_sigma(network) -> float:
    return small_worldness(network, sw_type="sigma")


def small_worldness_omega(network) -> float:
    return small_worldness(network, sw_type="omega")


def small_worldness(network: np.ndarray, sw_type: str = "sigma") -> float:
    # Convert the adjacency matrix to a binary matrix
    bin_matrix = network > 0

    # Compute the clustering coefficient and the shortest path length of the binary matrix
    c = np.mean(bct.clustering_coef_bu(bin_matrix.astype("uint8")))
    l = np.mean(bct.distance_bin(bin_matrix.astype("uint8")))

    # Compute the small world sigma
    rand_bin_matrix, _ = bct.randmio_und(bin_matrix, 10)
    c_r = np.mean(bct.clustering_coef_bu(rand_bin_matrix.astype("uint8")))
    l_r = np.mean(bct.distance_bin(rand_bin_matrix))

    if sw_type == "sigma":
        sw = (c / c_r) / (l / l_r)
    elif sw_type == "omega":
        raise NotImplementedError
        # c_l = compute_clustering_coef_lattice(network)
        # sw = (l / l_r) - (c / c_l)
    return sw


MEASURE_FUNCTIONS = {
    "global_efficiency_bin": global_efficiency_bin,
    "global_efficiency_wei": global_efficiency_wei,
    "assortativity_coeff_bin": assortativity_coeff_bin,
    "assortativity_coeff_wei": assortativity_coeff_wei,
    "average_degree": average_degree,
    "average_strength": average_strength,
    "density": density,
    "clustering_coeff_bin": clustering_coeff_bin,
    "clustering_coeff_wei": clustering_coeff_wei,
    "small_worldness_sigma": small_worldness_sigma,
    "small_worldness_omega": small_worldness_omega,
}


def compute_graph_measures(
    networks: np.ndarray,
    data: PetData,
    setup: Setup,
    output_path: str = None,
    measures_list: List[str] = MEASURES,
) -> pd.DataFrame:
    if output_path is None:
        out_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if output_path is None:
        output_path = os.path.join(out_dir, "graph_measures_python.csv")

    dim, n_samples, n_classes = np.shape(networks)
    dim = int(np.sqrt(dim))
    sample_indices = np.random.choice(n_samples, size=setup.n_samples_measures, replace=False)

    measures = []
    groups = []

    for c in tqdm(range(n_classes)):
        for s in sample_indices:
            net = np.reshape(networks[:, s, c], (dim, dim))
            measures.append(compute_global_gtm(net, measures_list))
            groups.append(data[c][1])

    df = pd.DataFrame(measures, columns=measures_list)
    df["groups"] = groups
    df.to_csv(output_path, index=False)

    return df


def compute_global_gtm(network: np.ndarray, measures: List[str], measures_func=MEASURE_FUNCTIONS) -> dict:
    measures_dict = {}

    for m in measures:
        if m in measures:
            measures_dict[m] = measures_func[m](network)

    return measures_dict
