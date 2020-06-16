import logging
import os
import pandas as pd
from pathlib import Path
from copy import deepcopy
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


def combine_and_add_valid(train_dfs, valid_dfs):
    if not isinstance(train_dfs, list):
        raise Exception("Please provide the training DataFrame(s) as a list")
    elif not isinstance(valid_dfs, list):
        raise Exception("Please provide the validation DataFrame(s) as a list")

    train_df = pd.concat(train_dfs)
    train_df["is_valid"] = False

    valid_df = pd.concat(valid_dfs)
    valid_df["is_valid"] = True

    return pd.concat([train_df, valid_df])


def format_databunch_df(df_to_update, folder_col_name,
                        leaf_col_name, create_copy=False):
    split_list = df_to_update[folder_col_name].str.split("/").to_list()
    counts_df = pd.DataFrame(split_list).apply(
        lambda x: len(x.unique()), axis=0)

    for i, count in enumerate(counts_df):
        if count > 1:
            break

    if create_copy:
        updated_df = deepcopy(df_to_update)
    else:
        updated_df = df_to_update

    updated_df[leaf_col_name] = updated_df.apply(
        lambda x: os.path.join(*str.split(x[folder_col_name], "/")[i:],
                               x[leaf_col_name]), axis=1)

    common_folder_path = os.path.join("/", *str.split(
        df_to_update[folder_col_name].to_list()[0], "/")[:i], "")

    updated_df.drop(folder_col_name, axis=1, inplace=True)

    return updated_df, common_folder_path


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
