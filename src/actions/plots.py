import logging
import sys
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def plot_embolism_profile(embolism_percentages: List[float],
                          leaf_name: str = None,
                          output_path: str = None,
                          show: bool = True,
                          **kwargs) -> None:
    """
     Plots an embolism profile. There are two plots. The first shows the
     VC for the series. The second plot shows the embolism percentage per
     mask. There is an option to save the plot.

    :param embolism_percentages: list of embolism percentages
    :param leaf_name: name of the leaf to use in the title
    :param output_path: path to save the plot, if None, the plot will not be
     saved
    :param show: whether the plot should be shown; if the plot is
     being saved, the user may not want to display the plot
    :param kwargs: subplot kwargs
    :return: None
    """
    cum_embolism_percentages = np.cumsum(embolism_percentages)

    if leaf_name:
        title = f"Embolism Profile of Leaf {leaf_name}"
    else:
        title = "Embolism Profile of Leaf"

    fig, axs = plt.subplots(3, **kwargs)
    fig.tight_layout(pad=8.0)
    fig.suptitle(title, fontsize=18, y=1)

    axs[0].plot(cum_embolism_percentages)
    axs[1].plot(embolism_percentages, color="orange")

    axs[0].set_xlabel('Steps', fontsize=14)
    axs[0].set_ylabel('% Embolism', fontsize=14)
    axs[0].set_title('Cumulative Embolism %', fontsize=16)

    axs[1].set_xlabel('Steps', fontsize=14)
    axs[1].set_ylabel('% Embolism', fontsize=14)
    axs[1].set_title('Total Embolism % per Mask', fontsize=16)

    if output_path:
        fig.savefig(output_path)

    if show:
        plt.show()


def plot_embolisms_per_leaf(summary_df: pd.DataFrame = None,
                            has_embolism_lol: List[List[float]] = None,
                            leaf_names_list: List[str] = None,
                            output_path: str = None,
                            show: bool = True,
                            percent: bool = False,
                            **kwargs) -> None:
    """
    Creates a bar plot showing the number of leaves with and without
    embolisms for a leaf. The input can either be a summary df, of list of
    embolism percentage lists.

    :param summary_df: a summary dataframe created using the code base; if this
     is None, then has_embolism_lol must be provided
    :param has_embolism_lol: list of embolism percentage lists; if this is
     None, then summary df must be provided
    :param leaf_names_list: the list of leaf names for the x-axis
    :param output_path: path to save the plot, if None, the plot will not be
     saved
    :param show: whether the plot should be shown; if the plot is
     being saved, the user may not want to display the plot
    :param percent: Annotate the bars using percentages; if this is false,
     counts will be used
    :param kwargs: subplot kwargs
    :return: None
    """
    leaf_names = []
    has_embolism_list = []

    if not summary_df and not has_embolism_lol:
        raise ValueError("Please provide either a summary_df or list of "
                         "has_embolism lists")

    if not summary_df:
        for i, h_e_list in enumerate(has_embolism_lol):
            if leaf_names_list:
                leaf_names = leaf_names + ([leaf_names_list[i]] *
                                           len(h_e_list))
            else:
                leaf_names = leaf_names + ([f"leaf {i}"] * len(h_e_list))

            has_embolism_list = has_embolism_list + h_e_list

        summary_df = pd.DataFrame({"has_embolism": has_embolism_list,
                                   "leaf": leaf_names})
    else:
        if len(summary_df.columns) != 2:
            raise ValueError(
                "A provided summary df should only have two columns, the "
                "first should contain whether or not a leaf has an embolism "
                "and the second should contain the leaf name")
        summary_df.columns = ["has_embolism", "leaf"]

    fig, ax = plt.subplots(**kwargs)

    temp_df = pd.DataFrame(summary_df.groupby("leaf")['has_embolism'].
                           value_counts(normalize=percent).round(2))

    temp_df.unstack().plot(kind='bar', ax=ax, rot=0)

    ax.legend(("False", "True"), title='Has Embolism', fontsize='large')
    ax.set_xlabel('Leaf', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Number of Images With Embolisms per Leaf', fontsize=16)

    rects = [rect for rect in ax.get_children() if
             isinstance(rect, mpl.patches.Rectangle)][:-1]

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    if output_path:
        fig.savefig(output_path)

    if show:
        plt.show()


# *============================ embolism profile =============================*
def plot_mseq_profiles(mseqs,
                       show: bool,
                       output_path_list: List[str],
                       leaf_name_list: List[str]) -> None:
    """
    Plots a sequence of embolism profiles using a list of MaskSequence objects.

    :param mseqs: list of MaskSequence objects to use
    :param show: whether the plot should be shown; if the plot is
     being saved, the user may not want to display the plot
    :param output_path_list: list of file paths to save the plot, if None, the
     plot will not be saved
    :param leaf_name_list: list of leaf names to use in the title of each plot
    :return: None
    """
    for i, mseq in enumerate(mseqs):
        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images(load_image=True)

        LOGGER.info("Extracting the intersection list")
        mseq.get_intersection_list()

        LOGGER.info("Extracting the embolism percent list")
        mseq.get_embolism_percent_list()

        if output_path_list is not None:
            mseq.plot_profile(show=show, output_path=output_path_list[i],
                              leaf_name=leaf_name_list[i], figsize=(10, 10))
        else:
            mseq.plot_profile(show=show, leaf_name=leaf_name_list[i],
                              figsize=(10, 10))

        mseq.unload_extracted_images()


# *============================ embolism bar plot ============================*
def plot_mseq_embolism_counts(mseqs,
                              show: bool,
                              output_path: str,
                              tiles: bool,
                              leaf_names_list: List[str],
                              leaf_embolism_only: bool = False,
                              percent: bool = False) -> None:
    """
    Creates a bar plot showing the number of leaves with and without
    embolisms for a leaf, using a list of MaskSequence objects

    :param mseqs: list of MaskSequence objects
    :param show: whether the plot should be shown; if the plot is
     being saved, the user may not want to display the plot
    :param output_path: path to save the plot, if None, the plot will not be
     saved
    :param tiles: whether the tiles belonging to the MaskSequence should be
     used
    :param leaf_names_list: the list of leaf names for the x-axis
    :param leaf_embolism_only: whether only leaves with embolisms should be
     counted
    :param percent: Annotate the bars using percentages; if this is false,
     counts will be used
    :return: None
    """

    has_embolism_lol = []

    for i, mseq in enumerate(mseqs):
        has_embolism_lol.append([])

        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images()

        if not tiles or leaf_embolism_only:
            mseq.load_image_array()

            LOGGER.info("Extracting the embolism percent list")
            mseq.get_embolism_percent_list()

            LOGGER.info("Extracting the has_embolism list")
            mseq.get_has_embolism_list()

            mseq.unload_extracted_images()

        if not tiles:
            has_embolism_lol[i] = has_embolism_lol[i] + mseq.has_embolism_list

        if tiles:
            with tqdm(total=len(mseq.image_objects), file=sys.stdout) as pbar:
                for mask_image in mseq.image_objects:
                    if leaf_embolism_only:
                        # Assumes linked sequences have been provided.
                        if not mask_image.has_embolism:
                            # Skip this image if the mask has no embolism
                            continue

                    mask_image.load_tile_paths()
                    mask_image.load_extracted_images(load_image=True,
                                                     disable_pb=True)
                    mask_image.get_embolism_percent_list(disable_pb=True)
                    mask_image.get_has_embolism_list(disable_pb=True)
                    has_embolism_lol[i] = \
                        has_embolism_lol[i] + mask_image.has_embolism_list
                    mask_image.unload_extracted_images()
                    pbar.update(1)

    if output_path is not None:
        plot_embolisms_per_leaf(has_embolism_lol=has_embolism_lol,
                                show=show, output_path=output_path,
                                leaf_names_list=leaf_names_list,
                                percent=percent, figsize=(15, 10))
    else:
        plot_embolisms_per_leaf(has_embolism_lol=has_embolism_lol,
                                show=show, leaf_names_list=leaf_names_list,
                                percent=percent, figsize=(15, 10))
# *===========================================================================*
