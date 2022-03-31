"""
extract_dataset.py
Script to extract leafs and masks to as model inputs
"""
__author__ = "Tristan Naidoo"
__maintainer__ = "Tristan Naidoo"
__version__ = "0.0.1"
__email__ = "ndxtri015@myuct.ac.za"
__status__ = "development"

import logging.config
from typing import Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# =============================================================================
# *------------------------- extracting image tiles --------------------------*
def chip_range(start: int,
               end: int,
               length: int,
               step: int = 1,
               overlap: bool = False) -> Tuple[int, int]:
    """
    A generator which is used to generate the start and end locations for a
    to extract a chip for a given image. E.g. for a image of size 1024x1024,
    the generator will return (0, 512), (512,1024). This is used for a
    single axis at a time.

    :param start: the starting location of the where to start tiling from;
     this is usually set to 0
    :param end: the end location where to stop chip; this is usually set to
     the length of the image at the relevant axis
    :param length: the length of the chip
    :param step: the size of the step to chip image.
    :param overlap: whether to overlap tiles when the tile size is larger
    than the portion of image remaining
    :return: (start pixel, end pixel)
    """
    current = start
    # Current + length is the upper end
    # (0,length) + current = current, current + length
    while (current + length) <= end:
        yield current, current + length
        current += step

    # In order to maintain equal length we reduce the step size
    rem_pixels = end - (current - step + length)

    if rem_pixels != 0:
        if overlap:
            # end - (last within bounds upper boundary)
            yield (current - step) + rem_pixels, end
        else:
            yield current, end


def chip_image(img: np.array,
               x_range: Tuple[int, int],
               y_range: Tuple[int, int]) -> np.array:
    """
    Chips a full size image, given tuples containing the x and y ranges of
    the region to be extracted.

    :param img: the input image as an np.array
    :param x_range: tuple containing the x region to chip
    :param y_range: tuple containing the y region to chip
    :return: an image chip
    """
    chip = img[y_range[0]:y_range[1], x_range[0]:x_range[1]]

    return chip


def pad_chip(img_chip: np.array,
             length_x: int,
             length_y: int,
             target_colour: int = 0) -> np.array:
    """
    Pads an image chip. The padding uses the target colour. This is
    generally the the colour of the background if padding for a segmentation
    task

    :param length_x: the target x length
    :param length_y: the target y length
    :param img_chip: the image chip to be padded
    :param target_colour: the colour to use when padding
    :return: a padded image chip
    """
    current_shape = img_chip.shape

    # need to take into account that we may need to pad in both directions
    if current_shape[0] < length_y:
        # pad y
        pixels_missing = (length_y - current_shape[0])
        img_chip = np.pad(img_chip, ((0, pixels_missing), (0, 0)),
                          constant_values=target_colour)  # pad bottom
    if current_shape[1] < length_x:
        # pad x
        pixels_missing = (length_x - current_shape[1])
        img_chip = np.pad(img_chip, ((0, 0), (0, pixels_missing)),
                          constant_values=target_colour)  # pad right

    return img_chip
# =============================================================================
