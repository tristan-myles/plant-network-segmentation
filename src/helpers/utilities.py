import logging
import os
from pathlib import Path
import matplotlib as mpl

LOGGER = logging.getLogger(__name__)


def create_file_name(output_folder_path, output_file_name, i,
                     placeholder_size):
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    file_name, extension = \
        str.rsplit(output_file_name, ".", 1)
    final_file_name = f"{file_name}_{i:0{placeholder_size}}.{extension}"
    final_file_name = os.path.join(output_folder_path, final_file_name)

    return final_file_name


def trim_image_array(image_array, output_size: int,
                     axis: str, trim_dir: int):
    """

    :param axis: one of either "x", "y" or both
    :param image_array:
    :param output_size:
    :param trim_dir:
    :return:
    """
    image_shape = image_array.shape

    if axis not in ["x", "y"]:
        raise ValueError("Please input one of 'x', 'y' as the axis")

    if trim_dir not in [-1, 1]:
        raise ValueError("Please input one of -1 or 1 as the trim "
                         "direction")

    if axis == "y":
        diff = image_shape[0] - output_size
    if axis == "x":
        diff = image_shape[1] - output_size
        image_array = image_array.transpose()

    if trim_dir == 1:
        # exclude the first x_diff columns
        image_array = image_array[diff:, :]
    if trim_dir == -1:
        # exclude the last x_diff rows
        image_array = image_array[:-diff, :]

    if axis == "x":
        image_array = image_array.transpose()

    return image_array


def update_plot_format(default: bool = False):
    if default:
        mpl.rcParams.update(mpl.rcParamsDefault)
    else:
        # Project specific params
        LOGGER.info(
            "Note these changes will apply the entire session, please run  "
            "this function with `default = True` to restore default "
            "matplotlib settings")
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
