import os
from typing import Optional
from .data_importer import PetData
from .base_classes import Setup

import bct
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def global_efficiency_bin(network: np.ndarray) -> float:
    """Calculate binary global efficiency."""
    return bct.efficiency_bin(network, local=False)


def global_efficiency_wei(network: np.ndarray) -> float:
    """Calculate weighted global efficiency."""
    return bct.efficiency_wei(network, local=False)


def assortativity_coeff_bin(network: np.ndarray) -> float:
    """Calculate binary assortativity coefficient."""
    return bct.assortativity_bin(network, flag=0)


def assortativity_coeff_wei(network: np.ndarray) -> float:
    """Calculate weighted assortativity coefficient."""
    return bct.assortativity_wei(network, flag=0)


def average_degree(network: np.ndarray) -> float:
    """Calculate average node degree."""
    return np.mean(bct.degrees_und(network))


def average_strength(network: np.ndarray) -> float:
    """Calculate average node strength."""
    return np.mean(bct.strengths_und(network))


def density(network: np.ndarray) -> float:
    """Calculate network density."""
    return bct.density_und(network)


def clustering_coeff_bin(network: np.ndarray) -> float:
    """Calculate binary clustering coefficient."""
    return np.mean(bct.clustering_coef_bu(network))


def clustering_coeff_wei(network: np.ndarray) -> float:
    """Calculate weighted clustering coefficient."""
    return np.mean(bct.clustering_coef_wu(network))


def small_worldness_sigma(network: np.ndarray) -> float:
    """Calculate small-worldness sigma."""
    return small_worldness(network, sw_type="sigma")


def small_worldness(network: np.ndarray, sw_type: str = "sigma") -> float:
    """
    Calculate small-worldness metrics.

    Args:
        network: Adjacency matrix
        sw_type: Type of small-worldness measure ('sigma' or 'omega')

    Returns:
        Small-worldness measure
    """
    # Convert to binary matrix
    bin_matrix = (network > 0).astype("uint8")

    # Compute clustering coefficient and path length
    c = np.mean(bct.clustering_coef_bu(bin_matrix))
    try:
        l = np.mean(bct.distance_bin(bin_matrix))
    except:
        return np.nan

    # Generate random network
    try:
        rand_bin_matrix, _ = bct.randmio_und(bin_matrix, 10)
        c_r = np.mean(bct.clustering_coef_bu(rand_bin_matrix))
        l_r = np.mean(bct.distance_bin(rand_bin_matrix))
    except:
        return np.nan

    if sw_type == "sigma":
        return (c / c_r) / (l / l_r)
    elif sw_type == "omega":
        raise NotImplementedError("Omega calculation not yet implemented")
    else:
        raise ValueError(f"Unknown small-worldness type: {sw_type}")


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
}


def compute_graph_measures(
    networks: np.ndarray,
    data: PetData,
    setup: Setup,
    output_path: Optional[str] = None,
    measures_list: list[str] = MEASURES,
) -> pd.DataFrame:
    """
    Compute graph theoretical measures for multiple networks.

    Args:
        networks: Array of shape (dim², n_samples, n_classes) containing adjacency matrices
        data: PetData object containing group information
        setup: Setup object containing configuration
        output_path: Path to save results CSV. If None, saves to ./outputs/
        measures_list: List of measures to compute

    Returns:
        DataFrame containing computed measures and group labels
    """

    # Handle output path
    if output_path is None:
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "graph_measures_python.csv")

    # Set random seed if provided
    if setup.seed is not None:
        np.random.seed(setup.seed)

    # Get dimensions
    total_nodes, n_samples, n_classes = networks.shape
    dim = int(np.sqrt(total_nodes))

    # Validate dimensions
    if dim * dim != total_nodes:
        raise ValueError(f"Network dimensions invalid: {total_nodes} is not a perfect square")

    # Validate measures
    invalid_measures = [m for m in measures_list if m not in MEASURE_FUNCTIONS]
    if invalid_measures:
        raise ValueError(f"Invalid measures specified: {invalid_measures}")

    # Sample indices if needed
    if setup.n_samples_measures < n_samples:
        sample_indices = np.random.choice(n_samples, size=setup.n_samples_measures, replace=False)
    else:
        sample_indices = range(n_samples)

    # Initialize results lists
    measures = []
    groups = []
    sample_ids = []
    class_ids = []

    # Compute measures for each network
    for class_idx in tqdm(range(n_classes), desc="Computing measures"):
        for sample_idx in sample_indices:
            # Reshape network to square matrix
            network = networks[:, sample_idx, class_idx].reshape(dim, dim)

            # Compute measures and store results
            network_measures = compute_global_gtm(network, measures_list)
            measures.append(network_measures)
            groups.append(data[class_idx][1])
            sample_ids.append(sample_idx)
            class_ids.append(class_idx)

    # Create DataFrame
    df = pd.DataFrame(measures)
    df["group"] = groups
    df["sample_id"] = sample_ids
    df["class_id"] = class_ids

    # Add metadata
    df.attrs["dim"] = dim
    df.attrs["n_samples"] = len(sample_indices)
    df.attrs["n_classes"] = n_classes

    # Save if path provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df


def compute_global_gtm(network: np.ndarray, measures: list[str], measures_func: dict = MEASURE_FUNCTIONS) -> dict:
    """
    Compute specified graph theoretical measures for a single network.

    Args:
        network: Square adjacency matrix
        measures: List of measure names to compute
        measures_func: Dictionary mapping measure names to their computing functions

    Returns:
        Dictionary of computed measures
    """
    measures_dict = {}

    for measure in measures:
        try:
            measures_dict[measure] = measures_func[measure](network)
        except Exception as e:
            print(f"Warning: Failed to compute {measure}: {str(e)}")
            measures_dict[measure] = np.nan

    return measures_dict
