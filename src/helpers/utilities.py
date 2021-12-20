import logging
import os
import sys
from pathlib import Path
from typing import List

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


# *================================= metrics =================================*
def get_iou_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates the Intersection over Union score given a mask and a prediction.

    :param y_true: the true mask corresponding to the prediction
    :param y_pred: the prediction
    :return: IoU
    """
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def classification_report(predictions: List[np.array],
                          masks: List[np.array],
                          save_path: str = None) -> pd.DataFrame:
    """
    Generates a classification report by comparing each mask and prediction
    in the input list of predictions and masks. If a save path is provided
    then the classification report is saved. The metrics returned in the
    report are: IoU, AUC_PR, Precision, Recall, F1, Accuracy, FN, FP, TN,
    and TP.

    :param predictions: a list of predictions
    :param masks: a list of masks corresponding to the predictions
    :param save_path: the output path to saved the classifcation report
    :return: a classification report df
    """
    metric_df = pd.DataFrame({"IoU": [], "Precision": [], "Recall": [],
                              "F1": [], "Accuracy": [], "AUC_PR": [],
                              "FN": [], "FP": [], "TN": [], "TP": []})

    with tqdm(total=len(predictions), file=sys.stdout) as pbar:
        for pred, mask in zip(predictions, masks):
            # grab the original image and reconstructed image
            y_true = mask.round().flatten()
            y_pred = pred.flatten()

            avg_pr = metrics.average_precision_score(y_true, y_pred)

            y_pred = y_pred.round()

            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred,
                                                      labels=[0,1]).ravel()
            iou = get_iou_score(y_true, y_pred)

            # set values to 0 if there are no true positives
            if tp > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = tp / (tp + 0.5 * (fn + fp))
            else:
                precision = 0
                recall = 0
                f1 = 0

            d = {"IoU": iou,
                 "AUC_PR": avg_pr,
                 "Precision": precision,
                 "Recall": recall,
                 "F1": f1,
                 "Accuracy": (tp + tn) / (tp + fn + tn + fp),
                 "FN": fn,
                 "FP": fp,
                 "TN": tn,
                 "TP": tp}

            metric_df = metric_df.append(d, ignore_index=True)

            pbar.update(1)

        if save_path:
            metric_df.to_csv(save_path)

    return metric_df


# *=============================== data model ================================*
def create_file_name(output_folder_path, output_file_name, i,
                     placeholder_size):
    """
    Creates the output folder path if it doesn't exist. The input number,
    i, is zero padded according to the placeholder size and appended to
    the output filename. This filename is appended to the output folder path
    and returned.

    :param output_folder_path: the output folder path
    :param output_file_name: the filename to which the zero-padded number
     must be appended
    :param i: the number to zero pad
    :param placeholder_size: determines how much to zero pad i; i.e. if
     placeholder size is 3 and i =1, the number used in the filename will be 001
    :return: an output file path
    """
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    file_name, extension = \
        str.rsplit(output_file_name, ".", 1)
    final_file_name = f"{file_name}_{i:0{placeholder_size}}.{extension}"
    final_file_name = os.path.join(output_folder_path, final_file_name)

    return final_file_name


def trim_image_array(image_array: np.array,
                     output_size: int,
                     axis: str,
                     trim_dir: int) -> np.array:
    """
    Trims an image, either from the start or the end, along a single axis.

    :param image_array: the input image
    :param output_size: the output size of the image; i.e. informs how much
     of the image to trim
    :param axis: one of either "x", "y" or both
    :param trim_dir: the trim direction, either -1 or 1; 1 trims from the
     start of the image and -1 trims from the end
    :return: the trimmed image
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


# *================================== plots ==================================*
def update_plot_format(default: bool = False) -> None:
    """
    Updates the matplotlib RC. This function can also be used to set the rc
    back to the default configuration.

    :param default: whether to set the RC to the custom config or to the
     default config
    :return: None
    """
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


# *================================= dataset =================================*
def create_subfolders(path: Path, folder: str) -> None:
    """
    Creates leaf and mask subfolders at the given base path and folder.

    :param path: base path
    :param folder: the folder in the path under which the subfolders should
     be created
    :return: None
    """
    path.joinpath(folder, "leaves").mkdir(parents=True,
                                          exist_ok=True)
    path.joinpath(folder, "masks").mkdir(parents=True,
                                         exist_ok=True)


# *===========================================================================*
