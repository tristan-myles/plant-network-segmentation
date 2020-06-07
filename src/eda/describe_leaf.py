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


def plot_embolism_profile(embolism_percentages, intersections,
                          output_path=None, show=True,
                          **kwargs):
    """
    :param embolism_percentages:
    :param intersections:
    :param kwargs:
    :return:
    """
    unique_embolism = np.array(embolism_percentages) - np.array(intersections)
    cum_embolism_percentages = np.cumsum(unique_embolism)

    fig, axs = plt.subplots(3, **kwargs)
    fig.tight_layout(pad=8.0)
    fig.suptitle('Embolism Profile of Leaf 1', fontsize=18, y=1)

    axs[0].plot(cum_embolism_percentages)
    axs[1].plot(unique_embolism, color="orange")

    axs[2].plot(embolism_percentages, color="orange")

    axs[0].set_xlabel('Steps', fontsize=14)
    axs[0].set_ylabel('% Embolism', fontsize=14)
    axs[0].set_title('Cummulative Embolism %', fontsize=16)

    axs[1].set_xlabel('Steps', fontsize=14)
    axs[1].set_ylabel('% Embolism', fontsize=14)
    axs[1].set_title('Total Embolism % per Mask', fontsize=16)

    axs[2].set_xlabel('Steps', fontsize=14)
    axs[2].set_ylabel('% Embolism', fontsize=14)
    axs[2].set_title('Unique Embolism % per Mask', fontsize=16)

    if show:
        plt.show()

    if output_path:
        plt.savefig(output_path)


if __name__ == "__main__":
    folder_path = "/mnt/disk3/thesis/data/1_qk3/tristan/"

    df = pd.read_csv(
        "/home/tristan/Documents/MSc_Dissertation/detecting-plant-network-failure/dataset_info.csv")
    df = df.drop(["idx", "idx.1", "Unnamed: 3"], axis=1)
    leaf_names = df[df.mask_types == "leaf_1"].mask_paths.apply(lambda x: x.rsplit("/", 1)[1])

    # Reading in images as np.arrays
    images = [np.array(PIL.Image.open(f"{folder_path}masks/{leaf_name}")) for
              leaf_name in leaf_names]

    # Unique range
    unique_range_list = list(map(get_unique_range, images))

    # Binarise images
    images = list(map(binarise_image, images))

    # Embolism percentages
    embolism_percentages = list(map(get_embolism_percent, images))

    # Extracting intersections
    combined_image = np.empty(shape=(images[0].shape[0], images[0].shape[1]),
                              dtype='object')

    intersection_list = []
    for image in images:
        intersection, combined_image = get_intersection(image, combined_image)
        intersection_list.append(intersection)

    # Viewing the results
    plot_embolism_profile(embolism_percentages, intersection_list,
                          figsize=(10, 15))

    # Saving the results
    output_df = pd.DataFrame({"name": leaf_names,
                              "unique_range": unique_range_list,
                              "embolism_percent": embolism_percentages,
                              "intersection:": intersection_list})

    output_df.to_csv("out.csv")
