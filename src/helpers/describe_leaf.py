"""
Script to provide descriptive information on images of leaves (in the project
format)
"""
__author__ = "Tristan Naidoo"
__maintainer__ = "Tristan Naidoo"
__version__ = "0.0.1"
__email__ = "ndxtri015@myuct.ac.za"
__status__ = "development"

import logging
from typing import List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def get_embolism_percent(image: np.array) -> int:
    """
    Returns the % of the image that are embolisms

    :param image: np.array of a mask
    :return: percentage of embolisms
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


def binarise_image(image: np.array,
                   lower_bound_255: int = 200,
                   upper_bound_0: int = 55) -> np.array:
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


def get_intersection(image: np.array,
                     combined_image: np.array) -> Tuple[int, np.array]:
    """
    Calculates the intersection between the current mask and all embolisms
    contained in previous masks

    :param image: np.array of a mask
    :param combined_image: np.array of a combined mask
    :return: intersection as a % of the image size and an updated
     combined image
    """

    intersection = np.count_nonzero((combined_image == 255) & (image == 255))
    intersection = (intersection / image.size)

    combined_image[image == 255] = 255

    return intersection, combined_image
