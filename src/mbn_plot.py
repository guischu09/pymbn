import itertools
import os
import shutil
import sys
from typing import Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from mayavi import mlab
from mne_connectivity.viz import plot_connectivity_circle
from skimage import measure

from .base_classes import Setup
from .data_importer import PetData
import logging


def get_output_dir(clobber: bool = False):
    output_dir = os.path.join(os.getcwd(), "outputs")
    if clobber and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_results_dir(clobber: bool = False):
    results_dir = os.path.join(os.getcwd(), "results")
    if clobber and os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


RESULTS_DIR = get_results_dir(clobber=False)
OUTPUT_DIR = get_output_dir(clobber=False)
NODE_SIZE_CTE = 1


class NetworkPlotter:
    def __init__(
        self,
        networks: np.ndarray,
        data: PetData,
        labels_path: str,
        atlas_path: str,
        coords_path: str,
        setup: Setup,
        clobber: bool = True,
    ) -> None:
        self.setup = setup
        self.networks = networks
        self.labels_path = labels_path
        self.atlas_path = atlas_path
        self.coords_path = coords_path
        self._set_output_dir(clobber)
        self._set_results_dir(clobber)
        self._set_group_names(data)

    def _set_group_names(self, data):
        group_names = []
        for d in data:
            group_names.append(d[1])
        self.group_names = group_names

    def _set_output_dir(self, clobber):
        output_dir = os.path.join(os.getcwd(), "outputs")
        if clobber and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        self.out_dir = output_dir

    def _set_results_dir(self, clobber):
        results_dir = os.path.join(os.getcwd(), "results")
        if clobber and os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        self.results_dir = results_dir

    def plot_networks_heatmaps(self, cmap: str = "turbo") -> None:
        self._set_vmin()

        for g, group in enumerate(self.group_names):
            network = self.networks[:, :, g]
            output_path = os.path.join(self.results_dir, f"{group}_heatmap.{self.setup.output_format}")
            logging.info(f">> Plotting heatmap for group {group}")
            plot_heatmap(network=network, labels=self.labels_path, output_path=output_path, v_min=self.v_min)
            logging.info(f">> Heatmap saved to {output_path}")

    def plot_networks_2d(
        self, facecolor: str = "white", textcolor: str = "black", cmap: str = "turbo", interactive: bool = False
    ) -> None:

        for g, group in enumerate(self.group_names):
            network = self.networks[:, :, g]
            network = network - np.eye(network.shape[0])
            output_path = os.path.join(self.results_dir, f"{group}_circle_plot.{self.setup.output_format}")
            labels = list(pd.read_csv(self.labels_path, header=None)[0])
            fig, _ = plot_connectivity_circle(
                con=network,
                node_names=labels,
                colormap=cmap,
                vmin=self.v_min,
                vmax=1,
                colorbar=True,
                facecolor=facecolor,
                textcolor=textcolor,
                show=interactive,
            )
            fig.savefig(output_path, facecolor=facecolor)
            plt.close()

    def plot_networks_brain3d(self, cmap: str = "turbo", interactive: bool = False):

        for g, group in enumerate(self.group_names):
            network = self.networks[:, :, g]
            output_path = os.path.join(self.results_dir, f"{group}_brain.{self.setup.output_format}")

            plot_3d(
                atlas=self.atlas_path,
                labels=self.labels_path,
                coordinates=self.coords_path,
                network=network,
                brain_type=self.setup.brain_type,
                output_path=output_path,
                v_min=self.v_min,
                cmap=cmap,
                interactive=interactive,
            )
        if interactive:
            sys.exit(0)

    def _set_vmin(self):
        unique_set = set(self.networks.ravel())
        unique_set.remove(0)
        self.v_min = min(unique_set)


def get_vmin(networks: np.ndarray) -> float:
    unique_set = set(networks.ravel())
    unique_set.remove(0)
    return min(unique_set)


def get_coords_dict(coords_path: str) -> dict:
    coords = pd.read_csv(coords_path, header=None)
    # Creates a dict where the keys are rois and values are x, y and z coordinates
    coordinates = {}
    for k in range(coords.shape[0]):
        coordinates[coords.iloc[k, 0]] = np.array([coords.iloc[k, 1], coords.iloc[k, 2], coords.iloc[k, 3]])
    return coordinates


def get_group_names(data: PetData) -> list[str]:
    group_names = []
    for d in data:
        group_names.append(d[1])
    return group_names


def plot_networks_2d(
    data: PetData,
    networks: np.ndarray,
    labels_path: str,
    output_format: str = ".png",
    interactive: bool = False,
    min_value: float = None,
    color_map: str = "turbo",
    results_dir: str = RESULTS_DIR,
    facecolor: str = "white",
    textcolor: str = "black",
) -> None:

    if min_value is None:
        min_value = get_vmin(networks)

    group_names = get_group_names(data)

    for g, group in enumerate(group_names):
        network = networks[:, :, g]
        network = network - np.eye(network.shape[0])
        output_path = os.path.join(results_dir, f"{group}_circle_plot.{output_format}")
        labels = list(pd.read_csv(labels_path, header=None)[0])
        fig, _ = plot_connectivity_circle(
            con=network,
            node_names=labels,
            colormap=color_map,
            vmin=min_value,
            vmax=1,
            colorbar=True,
            facecolor=facecolor,
            textcolor=textcolor,
            show=interactive,
        )
        fig.savefig(output_path, facecolor=facecolor)
        plt.close()


def plot_networks_heatmaps(
    data: PetData,
    networks: np.ndarray,
    labels_path: str,
    output_format: str = "png",  # png, pdf, svg
    color_map: str = "turbo",
    min_value: float = None,
    results_dir: str = RESULTS_DIR,
) -> None:

    if min_value is None:
        min_value = get_vmin(networks)

    group_names = get_group_names(data)

    for g, group in enumerate(group_names):
        network = networks[:, :, g]
        output_path = os.path.join(results_dir, f"{group}_heatmap.{output_format}")
        logging.info(f">> Plotting heatmap for group {group}")
        plot_heatmap(network=network, labels=labels_path, output_path=output_path, v_min=min_value, cmap=color_map)
        logging.info(f">> Heatmap image saved to {output_path}")


def plot_heatmap(
    network: Union[np.ndarray, str],
    labels: Union[list, str],
    output_path: str,
    v_min: float = 0,
    cmap: str = "turbo",
) -> None:
    if type(labels) == str:
        labels = list(pd.read_csv(labels, header=None)[0])
    elif type(labels) != list:
        raise ValueError("Non supported type. Supported types are List[str] and str.")

    if type(network) == np.ndarray:
        network = pd.DataFrame(network, columns=labels, index=labels)
    elif type(network) == str:
        network = pd.read_csv(network).values
    else:
        raise ValueError("Non supported type. Supported types are np.ndarray and str.")

    mask = np.isin(network, 0)

    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot heatmap figure. We believe that setting the labels in the x axis looks prettier.
    # Change xticklabels to True if you want to display the labels in the x axis of the heatmap as well.
    sns.heatmap(network, mask=mask, vmin=v_min, vmax=1, cmap=cmap, linecolor="k", linewidths=0.05, xticklabels=False)

    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    # Saves figure
    fig.savefig(output_path)

    plt.close()


def plot_2d(
    network_path: str,
    labels_path: str,
    output_path: str = None,
    facecolor: str = "white",
    textcolor: str = "black",
    cmap: str = "turbo",
    interactive: bool = None,
) -> None:
    if interactive is None:
        interactive = False

    network = pd.read_csv(network_path).iloc[:, 1::].values
    network = network - np.eye(network.shape[0])

    unique_set = set(network.ravel())
    unique_set.remove(0)
    v_min = min(unique_set)

    labels = list(pd.read_csv(labels_path, header=None)[0])

    if output_path is None:
        output_path = network_path.replace(".csv", "_circle.png")
    # norm = mpl.colors.Normalize(vmin=v_min, vmax=1)
    # m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, _ = plot_connectivity_circle(
        con=network,
        node_names=labels,
        colormap=cmap,
        vmin=v_min,
        vmax=1,
        colorbar=True,
        facecolor=facecolor,
        textcolor=textcolor,
        show=interactive,
    )
    fig.savefig(output_path, facecolor=facecolor)
    plt.close()


def plot_3d(
    atlas: Union[nib.minc1.Minc1Image, str],
    labels: Union[list, str],
    coordinates: Union[dict, str],
    network: Union[np.ndarray, str],
    brain_type: str,
    output_path: str = "brain.png",
    v_min: float = 0,
    cmap: str = "turbo",
    interactive: bool = False,
) -> None:
    if type(atlas) == str:
        atlas = nib.load(atlas)
    elif type(atlas) != nib.minc1.Minc1Image:
        raise ValueError("Non supported type. Supported types are nib.minc1.Minc1Image and str.")

    if type(labels) == str:
        labels = list(pd.read_csv(labels, header=None)[0])
    elif type(labels) != list:
        raise ValueError("Non supported type. Supported types are List[str] and str.")

    if type(coordinates) == str:
        coordinates = get_coords_dict(coordinates)
    elif type(coordinates) != dict:
        raise ValueError("Non supported type. Supported types are Dict and str.")

    if type(network) == np.ndarray:
        network = pd.DataFrame(network, columns=labels, index=labels)
    elif type(network) == str:
        network = pd.read_csv(network).values
    else:
        raise ValueError("Non supported type. Supported types are np.ndarray and str.")

    norm = mpl.colors.Normalize(vmin=v_min, vmax=1)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Obtain the non repeated pairs between rois
    pairs = np.array(list(itertools.combinations(labels, 2)))

    # Get Voxel Size:
    x_step = atlas.affine[0, 2]
    y_step = atlas.affine[1, 1]
    z_step = atlas.affine[2, 0]

    # Get Origin of image space
    x_start = atlas.affine[0, 3]
    y_start = atlas.affine[1, 3]
    z_start = atlas.affine[2, 3]

    # Get number os slices in the x plane
    x_dim = atlas.shape[2]

    # Get numpy array data and interchange axes to the X, Y, Z order
    brain = atlas.get_fdata()
    brain = np.swapaxes(brain, 0, 2)

    # discretize volume surface
    verts, faces, _, _ = measure.marching_cubes(
        brain, 10, spacing=(abs(x_step), abs(y_step), abs(z_step)), method="lorensen"
    )

    # Converts verts to three axis
    x, y, z = zip(*verts)

    # Adjusts coordinates to our atlas
    for k in coordinates.keys():
        coordinates[k][0] = coordinates[k][0] - (abs(x_start) + (abs(x_step) * -1) * x_dim)
        coordinates[k][1] = coordinates[k][1] - y_start
        coordinates[k][2] = coordinates[k][2] - z_start

    # Gets x, y and z coordinates of all rois
    Xn = [coordinates[k][0] for k in coordinates.keys()]  # x-coordinates of nodes
    Yn = [coordinates[k][1] for k in coordinates.keys()]  # y-coordinates
    Zn = [coordinates[k][2] for k in coordinates.keys()]  # z-coordinates

    # Size of nodes are associated with the node degree
    sizes = []
    strength = {}
    degree = {}
    for i, voi in enumerate(labels):
        sizes.append(np.sum(network.iloc[i] != 0) * 1.0)
        strength[voi] = np.sum(network.iloc[i])
        degree[voi] = np.sum(network.iloc[i] != 0) - 1

    adjSizes = []
    # Adjust sizes
    for i in range(len(labels)):
        adjSizes.append(sizes[i] ** 1.7 + NODE_SIZE_CTE)

    if brain_type == "mice/rat":
        NODE_MAX_SIZE = 0.65
    if brain_type == "human":
        NODE_MAX_SIZE = 8

    adjSizes = adjSizes / np.max(adjSizes) * NODE_MAX_SIZE

    ##### Mayvi scene ###
    if not interactive:
        mlab.options.offscreen = True

    # make a mayavi -mlab figure
    f = mlab.figure(1, size=(1000, 1000), fgcolor=(1, 1, 1), bgcolor=(1, 1, 1))

    # create and add a mesh brain to the figure
    mlab.triangular_mesh(x, y, z, faces, line_width=0.05, opacity=0.05, color=(0.82, 0.82, 0.82), figure=f)

    # Adjust the initial camera view. We found out that view(-55,54,71) looks well for rat brains and view()
    # view(180,90,546) looks decent for Humans

    if brain_type == "mice/rat":
        mlab.view(-55, 54, 71)
        rtube = 0.03

    if brain_type == "human":
        mlab.view(180, 90, 546)
        rtube = 0.08

    # Plot/Display link between nodes as lines using the information in the adjacency matrix.
    for p in pairs:
        correlation = network[p[0]][p[1]]

        if correlation != 0:
            xinit = coordinates[p[0]][0]
            yinit = coordinates[p[0]][1]
            zinit = coordinates[p[0]][2]

            xfinal = coordinates[p[1]][0]
            yfinal = coordinates[p[1]][1]
            zfinal = coordinates[p[1]][2]

            color_rgb = m.to_rgba(correlation)
            colors_rgb = color_rgb[0:3]

            mlab.plot3d(
                [xinit, xfinal],
                [yinit, yfinal],
                [zinit, zfinal],
                line_width=0.02,
                tube_radius=rtube,
                color=colors_rgb,
                figure=f,
            )

    # Creates the nodes on figure
    for i in range(len(labels)):
        s = adjSizes[i]
        if brain_type == "human":
            mlab.points3d(Xn[i], Yn[i], Zn[i], scale_factor=s, color=(0, 0, 0.58), line_width=0.02, opacity=1, figure=f)
        if brain_type == "mice/rat":
            mlab.points3d(Xn[i], Yn[i], Zn[i], scale_factor=s, color=(1, 1, 1), line_width=0.02, opacity=1, figure=f)

    if interactive:
        mlab.show()
    else:
        # Save figure
        mlab.savefig(output_path, magnification=10)
