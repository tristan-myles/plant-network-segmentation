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
import os
from glob import glob
from os import path
from pathlib import Path
from typing import List

import PIL.Image  # Addresses namespace conflict
import cv2
import numpy as np
from PIL import ImageChops, ImageSequence

LOGGER = logging.getLogger(__name__)

abs_path = path.dirname(path.abspath(__file__))
logging.config.fileConfig(fname=abs_path + "/../logging_configuration.ini",
                          defaults={'logfilename':
                                        abs_path + "/extract_dataset.log"},
                          disable_existing_loggers=False)
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
                    image = image / 255

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

            new_image = PIL.Image.open(input_path_list[i + dif_len_2])

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


def chip_range(start, end, step=1):
    """

    :param start:
    :param end:
    :param step:
    :return:
    """
    current = start

    while (current + step) < end:
        yield current, current + step
        current += step

    if end % step != 0:
        yield current, end


def chip_image(img, x_range, y_range):
    """

    :param img:
    :param x_range:
    :param y_range:
    :return:
    """
    chip = img[y_range[0]:y_range[1], x_range[0]:x_range[1]]

    return chip


def pad_chip(img_chip, stride_x, stride_y, target_colour=0):
    """

    :param stride_y:
    :param stride_x:
    :param img_chip:
    :param target_colour:
    :return:
    """
    current_shape = img_chip.shape

    # need to take into account that we may need to pad in both directions
    if current_shape[0] < stride_y:
        # pad y
        pixels_missing = (stride_y - current_shape[0])
        img_chip = np.pad(img_chip, ((0, pixels_missing), (0, 0)),
                          constant_values=target_colour)  # pad bottom
    if current_shape[1] < stride_x:
        # pad x
        pixels_missing = (stride_x - current_shape[1])
        img_chip = np.pad(img_chip, ((0, 0), (0, pixels_missing)),
                          constant_values=target_colour)  # pad right

    return img_chip


def image_to_chips(folder_path, leaf_name, mask_name, input_img, label_img,
                   stride_x=300, stride_y=300):
    Path(f"{folder_path}/diff-chips").mkdir(parents=True, exist_ok=True)
    Path(f"{folder_path}/mask-chips").mkdir(parents=True, exist_ok=True)

    input_ysize = input_img.shape[0]  # rows = y
    input_xsize = input_img.shape[1]  # cols = x

    counter = 0
    for y_range in chip_range(0, input_ysize, stride_y):
        for x_range in chip_range(0, input_xsize, stride_x):

            input_chip = chip_image(input_img, x_range, y_range)
            label_chip = chip_image(label_img, x_range, y_range)

            ychip_size = input_chip.shape[0]  # rows = y
            xchip_size = input_chip.shape[1]  # cols = x

            LOGGER.debug(f"{x_range} \t {y_range}")

            if ychip_size < stride_y or xchip_size < stride_x:
                input_chip = pad_chip(input_chip, stride_x, stride_y)
                label_chip = pad_chip(label_chip, stride_x, stride_y)

            new_leaf_name = leaf_name.rsplit("/", 1)[1].rsplit(".", 1)[0]
            new_mask_name = mask_name.rsplit("/", 1)[1].rsplit(".", 1)[0]

            input_chip_filename = os.path.join(folder_path,
                                               'diff-chips',
                                               f'{counter}_{new_leaf_name}.png')
            label_chip_filename = os.path.join(folder_path,
                                               'mask-chips',
                                               f'{counter}_{new_mask_name}.png')

            if not cv2.imwrite(input_chip_filename, input_chip):
                raise Exception("sigh")
            cv2.imwrite(label_chip_filename, label_chip)

            counter += 1


if __name__ == "__main__":
    common_path = "/mnt/disk3/thesis/data"

    folder_path_list = ["0_qk1_1", "1_qk3", "2_qtom1", "3_qgam1", "4_qp1",
                        "5_qp2", "6_qp4", "7_qp5"]

    mask_seq_name_list = ["Mask of Result of Substack (2-344).tif",
                          "Mask of Result of Substack (2-743).tif",
                          "Result of Substack (2-567).tif",
                          "Mask of Result of Substack (2-644).tif",
                          "Mask of Result of Substack (2-397).tif",
                          "Mask of Result of Substack (2-208).tif",
                          "Mask of Result of Substack (2-738).tif",
                          "Result of Substack (2-704).tif"]

    for i, (leaf_path, mask_seq_name) in enumerate(zip(folder_path_list,
                                                       mask_seq_name_list)):
        folder_path = f"{common_path}/{leaf_path}/"

        # Leaf extraction
        file_names = sorted([f for f in glob(folder_path + "2019*.tif",
                                             recursive=True)])
        diff_leaves_output_path = f"{folder_path}tristan/diffs/"
        diff_name = f"leaf_{i}_diff.tif"
        diff_path_list = extract_changed_sequence(
            file_names, diff_leaves_output_path, diff_name, overwrite=True)
        LOGGER.info(f"{len(diff_path_list)} difference images created")

        # Leaf extraction 2
        file_names = sorted([f for f in glob(folder_path + "2019*.tif",
                                             recursive=True)])
        diff_leaves_output_path = f"{folder_path}tristan/diffs_from_init/"
        diff_name = f"leaf_{i}_diff.tif"
        diff_path_list = extract_changed_sequence(
            file_names, diff_leaves_output_path, diff_name, dif_len=0,
            overwrite=True)
        LOGGER.info(f"{len(diff_path_list)} difference images created")

        # Mask extraction
        mask_output_folder_path = f"{folder_path}tristan/masks/"

        mask_path_list = extract_images_from_image_sequence(
            f"{folder_path}{mask_seq_name}", mask_output_folder_path,
            f"leaf_{i}_mask.png")
        LOGGER.info(f"{len(mask_path_list)} mask images created")

        # Binary extraction
        mask_output_folder_path = f"{folder_path}tristan/binary_masks/"

        mask_path_list = extract_images_from_image_sequence(
            f"{folder_path}{mask_seq_name}", mask_output_folder_path,
            f"leaf_{i}_binary_mask.png", binarise=True)
        LOGGER.info(f"{len(mask_path_list)} binary mask images created")

        # add masks
        file_names = [f for f in glob(f"{folder_path}tristan/masks/*.png",
                                      recursive=True)]
        file_names = sorted(file_names, key=lambda x: int(
            str.rsplit(str.rsplit(x, ".", 1)[0], "_", 1)[1]))

        diff_leaves_output_path = f"{folder_path}tristan/combined_masks/"
        diff_name = f"leaf_{i}_mask_overlay.png"
        diff_path_list = extract_changed_sequence(
            file_names, diff_leaves_output_path, diff_name, dif_len=0,
            combination_function=ImageChops.add_modulo,
            sequential=True,
            overwrite=True)
        LOGGER.info(f"{len(diff_path_list)} overlaid mask images created")
