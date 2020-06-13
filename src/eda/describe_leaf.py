"""
describe_leaf.py
Script to provide descriptive information on images of leaves (in the project
format)
"""
__author__ = "Tristan Naidoo"
__maintainer__ = "Tristan Naidoo"
__version__ = "0.0.1"
__email__ = "ndxtri015@myuct.ac.za"
__status__ = "development"

import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
from typing import List

LOGGER = logging.getLogger(__name__)


def get_embolism_percent(image: np.array) -> int:
    """
    Returns the % of the image that are emboli
    :param image: np.array of a mask
    :return: percentage of emboli
    """
    return np.count_nonzero(image == 255) / image.size


def get_unique_range(image: np.array) -> np.array:
    """
    Gets the unique list of pixel intensities for an image
    :param image: np.array of a mask
    :return: an array of unique pixel intensities
    """
    return np.unique(np.array(image))


def get_unique_leaf_range(images: List[np.array]) -> np.array:
    """
    Gets the unique list of pixel intensities from a list of images
    :param images: np.array of a mask
    :return: an array of unique pixel intensities
    """
    unique_range = np.array([])

    for i, image in enumerate(images):
        try:
            image_range = image
            pixel_ints_to_add = np.setdiff1d(image_range,
                                             unique_range)

            if pixel_ints_to_add.size != 0:
                unique_range = np.append(unique_range, pixel_ints_to_add)
        except TypeError as e:
            LOGGER.exception(f"image {i}  had an issue: \t", e)
            continue

    return unique_range


def binarise_image(image: np.array, lower_bound_255: int = 200,
                   upper_bound_0: int = 55) -> np.array :
    """
    Converts all pixels intensities within range of (lower_bound_255; 255)
    to 255 and all pixels intensities between (0; upper_bound_0) to 0. The aim
    is to binarise the image but this depends on the correct choice of boundary
    parameters.
    :param image: np.array of a mask
    :param lower_bound_255: lower bound of the range of values to be casted
    to 255
    :param upper_bound_0: upper bound of the range of values to be casted to 0
    :return: an np.array of a mask with only two pixel intensities: 0 and 255
    """
    image[(image > lower_bound_255) & (image < 255)] = 255
    image[(image > 0) & (image < upper_bound_0)] = 0

    return image


def get_intersection(image: np.array, combined_image: np.array) -> \
        (int, np.array):
    """
    Calculates the intersection between the current mask and all embolisms
    contained in previous masks
    :param image: np.array of a mask
    :param combined_image: np.array of a combined mask
    :return: the intersection as a % of the image size and an updated
    combined image
    """

    intersection = np.count_nonzero((combined_image == 255) & (image == 255))
    intersection = (intersection / image.size)

    combined_image[image == 255] = 255

    return intersection, combined_image


def plot_embolism_profile(embolism_percentages, intersections, leaf_name=None,
                          output_path=None, show=True, **kwargs):
    """
    :param embolism_percentages:
    :param intersections:
    :param kwargs:
    :return:
    """
    unique_embolism = np.array(embolism_percentages) - np.array(intersections)
    cum_embolism_percentages = np.cumsum(unique_embolism)

    if leaf_name:
        title = f"Embolism Profile of Leaf {leaf_name}"
    else:
        title = "Embolism Profile of Leaf"

    fig, axs = plt.subplots(3, **kwargs)
    fig.tight_layout(pad=8.0)
    fig.suptitle(title, fontsize=18, y=1)

    axs[0].plot(cum_embolism_percentages)
    axs[1].plot(unique_embolism, color="orange")

    axs[2].plot(embolism_percentages, color="orange")

    axs[0].set_xlabel('Steps', fontsize=14)
    axs[0].set_ylabel('% Embolism', fontsize=14)
    axs[0].set_title('Cumulative Embolism %', fontsize=16)

    axs[1].set_xlabel('Steps', fontsize=14)
    axs[1].set_ylabel('% Embolism', fontsize=14)
    axs[1].set_title('Total Embolism % per Mask', fontsize=16)

    axs[2].set_xlabel('Steps', fontsize=14)
    axs[2].set_ylabel('% Embolism', fontsize=14)
    axs[2].set_title('Unique Embolism % per Mask', fontsize=16)

    if output_path:
        fig.savefig(output_path)

    if show:
        plt.show()


def plot_embolisms_per_leaf(summary_df=None, has_embolism_lol=None,
                            leaf_names_list=None, output_path=None,
                            show=True, percent=False, **kwargs):
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
