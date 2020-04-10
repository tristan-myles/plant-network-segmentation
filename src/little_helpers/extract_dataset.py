"""
extract_dataset.py
Script to extract leafs and masks to as model inputs
"""
__author__ = "Tristan Naidoo"
__maintainer__ = "Tristan Naidoo"
__version__ = "0.0.1"
__email__ = "ndxtri015@myuct.ac.za"
__status__ = "development"

import os
from PIL import ImageChops, ImageSequence
import PIL.Image  # Addresses namespace conflict
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)


def extract_images_from_image_sequence(input_path: str,
                                       output_folder_path: str,
                                       output_name: str,
                                       overwrite: bool = False,
                                       binarise: bool = False) -> List[str]:
    """

    :param binarise:
    :param input_path:
    :param output_folder_path:
    :param output_name:
    :param overwrite:
    :return:
    """
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    output_path_list = []
    image_seq = PIL.Image.open(input_path)

    for (i, image) in enumerate(ImageSequence.Iterator(image_seq)):
        try:
            image = np.array(image)

            filename = str.rsplit(output_name, ".", 1)
            filename = f"{filename[0]}_{i}.{filename[1]}"
            filepath = os.path.join(output_folder_path, filename)

            create_file = False

            if not os.path.exists(filepath):
                create_file = True

            if overwrite:
                create_file = True

            if create_file:
                LOGGER.debug(f"Creating File: {filepath}")
                output_path_list.append(filepath)

                if binarise:
                    image = image/255

                cv2.imwrite(filepath, image)

        except FileNotFoundError as e:
            LOGGER.exception("Strange :? \n", e)
            break

    return output_path_list


def extract_changed_sequence(input_path_list: str,
                             output_folder_path: str,
                             output_name: str,
                             dif_len: int = 1,
                             combination_function=ImageChops.subtract_modulo,
                             sequential: bool = False,
                             overwrite: bool = False) -> List[str]:
    """

    :param sequential:
    :param combination_function:
    :param dif_len:
    :param input_path_list:
    :param output_folder_path:
    :param output_name:
    :param overwrite:
    :return:
    """
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    output_path_list = []

    if dif_len == 0:
        old_image = PIL.Image.open(input_path_list[0])
        dif_len_2 = 1
    else:
        dif_len_2 = dif_len

    for i in range(0, (len(input_path_list) - dif_len_2)):
        create_file = False

        try:
            if dif_len != 0:
                old_image = PIL.Image.open(input_path_list[i])

            new_image = PIL.Image.open(input_path_list[i+dif_len_2])

            combined_im = combination_function(new_image, old_image)

            filename = str.rsplit(output_name, ".", 1)
            filename = f"{filename[0]}_{i}.{filename[1]}"
            filepath = os.path.join(output_folder_path, filename)

            if not os.path.exists(filepath):
                create_file = True

            if overwrite:
                create_file = True

            if create_file:
                LOGGER.debug(f"Creating File: {filepath}")
                output_path_list.append(filepath)
                with open(filepath, "w") as f:
                    combined_im.save(filepath)

        except FileNotFoundError as e:
            LOGGER.exception("Strange :? \n", e)
            break

        if sequential:
            old_image = combined_im

    return output_path_list
