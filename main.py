import numpy as np

from src.data_importer import data_import
from src.mbn_builder import build_network
from src.mbn_measures import compute_graph_measures
from src.ui_parser import ManualSetup as SetupParser
from src.mbn_plot import plot_networks_heatmaps, plot_networks_2d, plot_networks_3d, NetworkPlotter


def main():
    # Set input parameters
    setup = SetupParser().get_parameters()
    np.random.seed(setup.seed)

    # Load input data
    data, labels, atlas, coords = data_import(setup)

    # Build MBNs
    group_networks, multiple_networks = build_network(data, setup)

    # Generate and save heatmaps:
    plot_networks_heatmaps(data, group_networks, labels, output_format="svg")

    # Generate and save circle plots:
    plot_networks_2d(data, group_networks, labels, output_format="svg", interactive=False)

    # Generate and save 3d brain plots:
    plot_networks_3d(
        data,
        group_networks,
        labels,
        atlas,
        coords,
        brain_type=setup.brain_type,
        output_format="png",
        interactive=True,
    )


if __name__ == "__main__":
    main()
