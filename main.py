import numpy as np

from src.data_importer import data_import
from src.mbn_builder import build_network
from src.mbn_measures import compute_graph_measures
from src.ui_parser import ManualSetup as SetupParser
from src.mbn_plot import plot_networks_heatmaps, plot_networks_2d


def main():
    # Set input parameters
    setup = SetupParser().get_parameters()
    np.random.seed(setup.seed)

    # Load input data
    data, labels, atlas, coords = data_import(setup)

    # Build MBNs
    group_networks, multiple_networks = build_network(data, setup)

    # Graph measures
    compute_graph_measures(multiple_networks, data, setup)

    # Generate and save plots:
    plot_networks_heatmaps(data, group_networks, labels, output_format="svg")

    plot_networks_2d(data, group_networks, labels, output_format="svg", interactive=True)

    # # Make plots
    # plotter = NetworkPlotter(
    #     networks=group_networks,
    #     data=data,
    #     labels_path=labels,
    #     atlas_path=atlas,
    #     coords_path=coords,
    #     setup=setup,
    # )

    # plotter.plot_networks_heatmaps()
    # plotter.plot_networks_2d(interactive=True)
    # plotter.plot_networks_brain3d(interactive=False)


if __name__ == "__main__":
    main()
