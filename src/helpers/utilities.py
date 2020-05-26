import os
import pandas as pd
from pathlib import Path
from copy import deepcopy


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
