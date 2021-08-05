import argparse
import json
import logging
import os
import pprint
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import kerastuner
import numpy as np
import tensorflow as tf
from kerastuner import Objective
from kerastuner.tuners import BayesianOptimization
from sklearn.metrics import precision_recall_curve

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# *================================= general =================================*
def save_prcurve_csv(run_name: str,
                     mask: List[np.array],
                     pred: List[np.array],
                     type: str) -> None:
    """
    Saves the result of a precision recall curve as a json. The elements are
    precision, recall, and the threshold. If there are more than 1000000
    thresholds, the elements are downsampled to 1000 items.

    :param run_name: the run name to use when saving the file
    :param mask: the masks
    :param pred: the predictions
    :param type: the type of prcurve to be used in the save name; e.g. test
     or val
    :return: None
    """
    y_true = np.array(mask).flatten()
    y_pred = np.array(pred).flatten()

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    if len(thresholds) > 1000000:
        reduced_p = []
        reduced_r = []
        reduced_t = []

        for i in range(1, len(thresholds), 1000):
            reduced_p.append(precision[i])
            reduced_r.append(recall[i])
            reduced_t.append(thresholds[i])

    else:
        reduced_p = precision
        reduced_r = recall
        reduced_t = threshold

    pr_json = {"precision": list(np.array(reduced_p).astype("str")),
               "recall": list(np.array(reduced_r).astype("str")),
               "thresholds": list(np.array(reduced_t).astype("str"))}

    # Output path is handled differently compared to rest of file
    project_root = Path(os.getcwd())
    output_path = project_root.joinpath("data", "run_data",
                                        "pr_curves")
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path.joinpath(run_name + "_" + type + "_pr_curve.json")

    with open(filename, "w") as file:
        json.dump(pr_json, file)


def save_predictions(prediction: np.array,
                     folder_path: str,
                     filename: str) -> None:
    """
    Saves a tf model prediction

    :param prediction: the tf model prediction
    :param folder_path: the folder path to save the image
    :param filename: the filename to save the image
    :return: None
    """
    output_folder_path = os.path.join(folder_path, "../predictions")
    folder_path_1, folder_path_2 = filename.rsplit("/", 4)[1:3]
    output_folder_path = os.path.join(output_folder_path, folder_path_1,
                                      folder_path_2)
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    filename = filename.rsplit("/", 1)[1]
    filepath = os.path.join(output_folder_path, filename)

    cv2.imwrite(filepath, prediction)


def save_compilation_dict(answers: Dict, lr: float, save_path: str) -> None:
    """
    Saves the a json of the compilation used for a training run.

    :param answers: a dictionary of answers which contains the user choices
      for the model, loss, optimiser, filter multiple, loss weight,
      initialiser, and activation
    :param lr: the learning rate
    :param save_path: where to save the json
    :return: None
    """
    model_choices = ["unet", "unet_resnet", "wnet"]
    loss_choices = ["bce", "wce", "focal", "dice"]
    opt_choices = ["adam", "sgd"]

    compilation_dict = {"model": model_choices[answers["model_choice"]],
                        "loss": loss_choices[answers["loss_choice"]],
                        "opt": opt_choices[answers["opt_choice"]],
                        "filters": answers["filters"],
                        "loss_weight": answers["loss_weight"],
                        "initializer": answers["initializer"],
                        "activation": answers["activation"],
                        "lr": str(lr)}

    save_path = save_path + "compilation.json"
    with open(save_path, 'w') as file:
        json.dump(compilation_dict, file)


def get_sorted_list(search_string: str) -> List[str]:
    """
    Returns a sorted list of file paths given a search string.

    :param search_string: the search string to use; this should be a
     complete path with regex expressions for filenames
    :return: list of file paths
    """
    return sorted([str(f) for f in glob(search_string, recursive=True)])


def configure_for_performance(ds: tf.data.Dataset,
                              batch_size: int = 2,
                              buffer_size: int = 100) -> None:
    """
    Shuffles a tf dataset and batches it. It also configures the amount of
    batches to prefetch which improves performance.

    :param ds: the tf dataset
    :param batch_size: the batch size
    :param buffer_size: the buffer size; this controls the extent that the
      dataset will be shuffled
    :return: tf dataset configured for performance
    """
    # ds.cache() speeds up performance but uses too much memory in this project
    # ds = ds.cache()
    # Number of batches to use when shuffling (calls x examples sequentially
    # => bigger buffer = better shuffle)
    if buffer_size:
        ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size)

    # number of batches to prefetch
    ds = ds.prefetch(buffer_size=batch_size)
    return ds


def tune_model(hyperModel: kerastuner.HyperModel,
               train_dataset: tf.data.Dataset,
               val_dataset: tf.data.Dataset,
               results_dir: str,
               run_name: int,
               input_shape: Tuple[int],
               output_channels: int = 1) -> None:
    """
    Performs Bayesian hyperparameter optimization using the input HyperModel.

    :param hyperModel: a kerastuner HyperModel
    :param train_dataset: the training dataset to use in the optimization
    :param val_dataset: the validation set  to use in the optimization
    :param results_dir: where the results of the optimisation should be stored
    :param run_name: the run name, which is used when saving the results
    :param input_shape: the input shape of HyperModel input
    :param output_channels: the number of output channels of the HyperModel
     output
    :return: None
    """
    model = hyperModel(input_shape, output_channels)

    tuner = BayesianOptimization(
        model,
        objective=Objective('val_auc', direction="max"),
        max_trials=30,
        directory=results_dir,
        project_name=run_name,
        seed=3141)

    tuner.search(
        x=train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    min_delta=0.000001)])

    tuner.search_space_summary()

    tuner.results_summary()


def save_lrt_results(lr_range_test,
                     save_path: str) -> None:
    """
    Saves the result of a learning rate range test as a json.

    :param lr_range_test: a LRRangeTest object
    :param save_path: where to save the json
    :return: None
    """
    lrt_dict = {"smooth loss": lr_range_test.smoothed_losses,
                "batch num": lr_range_test.batch_nums,
                "log lr": lr_range_test.batch_log_lr}
    save_path = save_path + "_lrt_results.json"
    with open(save_path, 'w') as file:
        json.dump(lrt_dict, file)


def get_class_weight(training_path: str, incl_aug: bool) -> float:
    """
    Gets the % of embolism pixels to use as a class weight in a weighted
    loss function

    :param training_path: the path to the training images
    :param incl_aug: whether augmented images should be included
    :return: the % of embolism pixels
    """
    embolism_pixels = 0
    total_pixels = 0

    if incl_aug:
        search_string = '*/masks/*.png'
    else:
        search_string = "*embolism/masks/*.png"

    for path in Path(training_path).rglob(search_string):
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        embolism_pixels += np.sum(im == 255)
        total_pixels += im.size

    class_pers = embolism_pixels / total_pixels
    LOGGER.info(f"% of embolism pixels: {round(class_pers * 100, 2)}%")

    return class_pers


# *============================ image processing =============================*
# *----------------------------- pre-processing ------------------------------*
def global_contrast_normalization(img: tf.Tensor,
                                  s: int = 1,
                                  lamb: int = 10,
                                  eps: float = 10e-8) -> tf.Tensor:
    """
    Applies global contrast normalisation, which aims to standardise the
    contrast across pixels in the image.

    :param img: input image
    :param s: scale parameter; the standard deviation across all pixels is
     equal to s
    :param lamb: lambda, which is regularisation parameter
    :param eps: episilon, the minimum size of the denomintor; this can be used
     instead of lambda
    :return: a normalized image
    """
    total_mean = tf.math.reduce_mean(img)
    contrast = tf.math.reduce_mean(tf.math.pow(img - total_mean, 2))
    biased_contrast = tf.math.sqrt(lamb + contrast)

    denom = tf.math.maximum(biased_contrast, eps)

    normalized_image = tf.math.scalar_mul(s, (
        tf.math.divide((img - total_mean), denom)))

    return normalized_image


# *------------------------------ image parsing ------------------------------*
def read_file(img_path: str,
              shift_256: bool,
              transform_uint8: bool) -> tf.Tensor:
    """
    Reads in an image from the file path and converts it to a tf tensor.

    :param img_path: file path of the image to read in
    :param shift_256: whether to shift the image by 256
    :param transform_uint8: whether to transform the image to a uint8 format
    :return: an image as tf.float32 tensor
    """
    img = cv2.imread(img_path.decode("utf-8"), cv2.IMREAD_UNCHANGED)

    if shift_256:
        # if the image was shifted by 256 when saved, then shift back to
        # restore negative values
        img = img.astype(np.int16) - 256
    elif transform_uint8:
        # if a shifted image was provided convert back to a uint8 to view
        # note, can't convert back
        img = img.astype(np.uint8)

    img = tf.convert_to_tensor(img, dtype=tf.float32)

    return img


def parse_numpy_image(img: np.array,
                      batch_shape: int) -> tf.Tensor:
    """
    Converts an np.array to tf.float32 tensor, normalises the image by
    dividing by 255, and reshapes the image according to the batch shape.

    :param img: the image to be parsed
    :param batch_shape: the batch shape
    :return: a parsed image as a tf.float32 tensor
    """
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, batch_shape)

    return img


def parse_image_fc(leaf_shape: Tuple[int, int, int],
                   mask_shape: Tuple[int, int, int],
                   test: bool = False,
                   shift_256: bool = False,
                   transform_uint8: bool = False):
    """
    A function closure which creates leaf, mask samples. This is used as a
    preparation step for a tf.Dataset.

    :param leaf_shape: the shape of the leaf
    :param mask_shape: the shape of the leaf
    :param test: whether test set samples are being parsed (currently this is
     not used)
    :param shift_256: whether to shift the image by 256
    :param transform_uint8: whether to transform the image to a uint8 format
    :return: the inner function which expects an leaf path and mask path
    """

    def parse_image(img_path: str, mask_path: str):
        # load the raw data from the file as a string
        img = tf.numpy_function(read_file,
                                [img_path, shift_256, transform_uint8],
                                tf.float32)
        img = tf.where(tf.greater_equal(img, tf.cast(0, tf.float32)),
                       tf.cast(0, tf.float32), img)
        img = tf.reshape(img, leaf_shape)

        # Applying median filter
        # img = tfa.image.median_filter2d(img)
        # img = global_contrast_normalization(img)

        # Masks
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.reshape(mask, mask_shape)

        # if not test:
        #     mask = tfa.image.gaussian_filter2d(mask)
        #     # will return 255 where the condition is met else 0
        #     mask = tf.where(tf.greater(mask, 0), 255, 0)

        mask = tf.cast(mask, tf.float32) / 255.0
        img = tf.cast(img, tf.float32) / 255.0

        return img, mask

    return parse_image


# *----------------------------- post-processing -----------------------------*
def im2_lt_im1(pred: np.array, input_image: np.array) -> np.array:
    """
    Converts all pixels in the prediction to 0 in locations where the input
    image pixels are positive

    :param pred: the prediction to process
    :param input_image: the input image corresponding the prediction
    :return: a processed prediction
    """
    # reducing false positives, i.e trying to improve precision
    pred[(input_image >= 0)] = 0

    return pred


def gaussian_blur(img: np.array,
                  kernel: Tuple[int] = (5, 5),
                  base: int = 1) -> np.array:
    """
    Applies a Gaussain blur to an image. Pixels which are greater than zero
    are set the base pixel provided. This is intended to be used to blur
    masks and predictions.

    :param img: the image to blur
    :param kernel: Gaussian kernel, the first element is kernel height and
     the second element is kernel width; both elements must be positive and odd
    :param base: the value to set pixels greater than zero to, after blurring
    :return: a blurred image
    """
    # reducing false positives, i.e trying to improve precision
    img = cv2.GaussianBlur(img.astype(np.float32), kernel, 1)
    img[img > 0] = base
    img = img.astype(np.uint8)
    return img


def threshold(img: np.array, thresh: float) -> np.array:
    """
    Thresholds a prediction, by converting all pixels greater than the
    threshold to 1 and the remainder to 0.

    :param img: the image prediction to threshold
    :param thresh: the threshold
    :return: a prediction with pixel intensities of 0 or 1
    """
    img[img >= thresh] = 1
    img[img < thresh] = 0

    return img


# *=============================== load model ================================*
def check_model_save(model: tf.keras.Model,
                     new_model: tf.keras.Model,
                     new_loss: tf.keras.losses.Loss,
                     new_opt: tf.keras.optimizers.Optimizer,
                     answers: Dict,
                     metrics: List[tf.keras.metrics.Metric],
                     model_save_path: str,
                     check_opt=False) -> None:
    """
    Checks if a model have been saved correctly.

    :param model: the old model to compare to
    :param new_model: the saved model
    :param new_loss: an instance of tf loss to attach to the new model
    :param new_opt: an instance of tf optimiser to attach to the new model;
     this must be instantiated with the correct learning rate
    :param answers: the answers from the input prompt used in the training run
    :param metrics: the metrics to attach to the model
    :param model_save_path: the save path of the model (where the weights
     are located)
    :param check_opt: whether the optimiser state should be checked
    :return: None
    """
    old_pred, old_bloss = get_model_pred_batch_loss(
        model, answers["leaf_shape"], answers["mask_shape"])

    new_opt = new_opt(model.optimizer.lr.numpy())
    print("warning deleting model")
    del model

    new_model.load_workaround(answers["leaf_shape"], answers["mask_shape"],
                              new_loss, new_opt, metrics, model_save_path)

    new_pred, new_bloss = get_model_pred_batch_loss(
        new_model, answers["leaf_shape"], answers["mask_shape"])

    np.testing.assert_allclose(old_pred, new_pred, atol=1e-6,
                               err_msg="Prediction from the saved model is"
                                       " not the same as the original model")

    if check_opt:
        assert old_bloss == new_bloss, "Optimiser state not preserved!"


def get_model_pred_batch_loss(model: tf.keras.Model,
                              leaf_shape: Tuple[int, int, int],
                              mask_shape: Tuple[int, int, int]) -> \
        Tuple[np.array, float]:
    """
    Gets the prediction and batch loss for an input and mask of zeros.

    :param model: the model to make a prediction with
    :param leaf_shape: the leaf shape
    :param mask_shape: the leaf shape
    :return: the prediction and the batch loss
    """
    # Create a blank input image and mask
    x_train_blank = np.zeros((1,) + leaf_shape)
    y_train_blank = np.zeros((1,) + mask_shape)

    # Check that the model state has been preserved
    predictions = model.predict(x_train_blank)

    # Checking that the optimizer state has been preserved
    # Seems to work when these are repeated twice - not sure why?
    batch_loss = model.train_on_batch(x_train_blank, y_train_blank)
    batch_loss = model.train_on_batch(x_train_blank, y_train_blank)

    return predictions, batch_loss


# *=============================== tf __main__ ===============================*
# *-------------------------------- argparse ---------------------------------*
def parse_arguments() -> argparse.Namespace:
    """
     Argument parser

    :return: An argparse namespace
    """
    parser = argparse.ArgumentParser("Perform operations using the "
                                     "plant-image-segmentation code base")

    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Whether to run this script in interactive mode, where the user "
             "will be guided through the required input")

    parser.add_argument("--json_path", "-jp", type=str,
                        help="path to a JSON file with the inputs required to"
                             "train a tf model using this codebase")

    args = parser.parse_args()

    return args


# *--------------------------- interactive prompt ----------------------------*
def print_options_dict(output_dict):
    """
    Print a formatted version of the results of the output_dict generated by
    the input prompt

    :param output_dict: the output dict to print
    :return: None
    """
    print(f"\nYour chosen configuration is:\n")
    for i, (key, val) in enumerate(output_dict.items()):
        if i == 0:
            num = "  "
        elif i == 1:
            num = str(i) + "."
        elif i == 2 or i == 3:
            num = "  "
        else:
            num = str(i - 2) + "."

        if isinstance(val, dict):
            print_str = "\n" + pprint.pformat(val) + "\n"
        else:
            print_str = val

        print(f"{num} {(' '.join(key.split('_'))).capitalize() :<40}:"
              f" {print_str}")


def interactive_prompt() -> None:
    """
    Interactive prompt which allows users to interact with the code base.
    The results are returned as a dict. There is an option, through the
    prompt, to save the input as a json to be used again.

    :return: None
    """
    happy = False
    options_list = {-1}

    print("Hello and welcome to plant-network-segmentation's training module"
          "\nA few inputs will be required from you to begin...")

    while not happy:
        if -1 in options_list:
            output_dict = {}

            print("* What action would you like to take?\n"
                  "-------------------------------------\n"
                  " 1. Tuning\n"
                  " 2. Training"
                  )

            operation = int(input("Please select your operation: "))
            output_dict["which"] = ["tuning", "training"][operation - 1]

            # list options = 1 - 18
            options_list.update(range(1, 24))

            options_list.remove(-1)

        if 1 in options_list:
            train_base_dir = input("\n 1. Where is the base directory of your"
                                   " training images?\nNote, please include a"
                                   " / at the end of the directory: ")

            output_dict["train_base_dir"] = train_base_dir

            options_list.remove(1)

            print("\nWhat format should the images be loaded in?\n"
                  "Note: this also applies to validation and test images\n"
                  "   Options:\n"
                  "   0: uint8\n"
                  "   1: shifted -256 (don't chose this option unless "
                  "images were shifted by +256 when saved)")

            leaves_format = input("Please choose a number: ")

            if leaves_format == "":
                leaves_format = None
            else:
                leaves_format = int(leaves_format)

            transform_uint8 = False
            shift_256 = False

            if leaves_format == 0:
                transform_uint8 = True
            elif leaves_format == 1:
                shift_256 = True

            output_dict["transform_uint8"] = transform_uint8
            output_dict["shift_256"] = shift_256

        if 2 in options_list:
            val_base_dir = input("\n 2. Where is the base directory of your"
                                 " validation images?\nNote, please include a"
                                 " / at the end of the directory: ")

            output_dict["val_base_dir"] = val_base_dir

            options_list.remove(2)

        if 3 in options_list:
            leaf_ext = input("\n 3. What is the leaf extension: ")

            output_dict["leaf_ext"] = leaf_ext

            options_list.remove(3)

        if 4 in options_list:
            mask_ext = input(
                "\n 4. What is the mask extension: ")
            output_dict["mask_ext"] = mask_ext

            options_list.remove(4)

        if 5 in options_list:
            print("\n 5. Should augmented images be included in the training "
                  "set?\n"
                  "Options:\n"
                  "0: False\n"
                  "1: True")
            incl_aug = input("Please choose a number: ")
            incl_aug = int(incl_aug) == 1

            output_dict["incl_aug"] = incl_aug

            options_list.remove(5)

        if 6 in options_list:
            leaf_shape = input("\n 6. Please enter the leaf image shape, "
                               "separating"
                               " each number by a space: ")

            output_dict["leaf_shape"] = tuple(
                [int(size) for size in leaf_shape.split()])

            options_list.remove(6)

        if 7 in options_list:
            mask_shape = input("\n 7. Please enter the mask image shape, "
                               "separating"
                               " each number by a space: ")

            output_dict["mask_shape"] = tuple(
                [int(size) for size in mask_shape.split()])

            options_list.remove(7)

        if 8 in options_list:
            print("\n 8. Please choose which model you would like to use\n"
                  "Options:\n"
                  "0: Vanilla U-Net\n"
                  "1: U-Net with ResNet backbone \n"
                  "2: W-Net\n")
            model_choice = int(input("Please choose the relevant model"
                                     " number: "))

            output_dict["model_choice"] = int(model_choice)

            options_list.remove(8)

        if 9 in options_list:
            buffer_size = int(input(
                "\n 9. Please provide a buffer size\nNote,"
                " this influences the extent to which the data is "
                "shuffled: "))

            output_dict["buffer_size"] = int(buffer_size)

            options_list.remove(9)

        if 10 in options_list:
            batch_size = int(input("\n10. Please provide a batch size: "))

            output_dict["batch_size"] = int(batch_size)

            options_list.remove(10)

        if operation == 1:
            if 11 in options_list:
                run_name = input(
                    "\n11. Please enter the run name, this will be the name "
                    "used to save your hyperparameter tuning output: ")

                output_dict["run_name"] = run_name

                options_list.remove(11)

        if operation == 2:
            if 11 in options_list:
                print("\n11. Please choose which loss function you would like"
                      " to use\n"
                      "Options:\n"
                      "0: Balance cross-entropy\n"
                      "1: Weighted cross-entropy \n"
                      "2: Focal loss\n"
                      "3: Soft dice loss\n")
                loss_choice = int(input("\nPlease choose the relevant loss"
                                        " function number: "))

                output_dict["loss_choice"] = int(loss_choice)

                options_list.remove(11)

            if 12 in options_list:
                print("\n12. Please choose which optimiser you would like"
                      " to use\n"
                      "Options:\n"
                      "0: Adam\n"
                      "1: SGD with momentum \n")
                opt_choice = int(input("Please choose the relevant optimiser"
                                       " number: "))

                output_dict["opt_choice"] = int(opt_choice)

                options_list.remove(12)

            if 13 in options_list:
                lr = float(input("\n13. Please provide a learning rate: "))

                output_dict["lr"] = float(lr)

                options_list.remove(13)

            if 14 in options_list:
                epochs = float(input(
                    "\n14. Please provide the number of epochs to run for: "))

                output_dict["epochs"] = int(epochs)

                options_list.remove(14)

            if 15 in options_list:
                print("\n15. Please choose which callbacks you would like"
                      " to use\n"
                      "Options:\n"
                      "0: Learning rate range test\n"
                      "1: 1cycle policy\n"
                      "2: Early stopping\n"
                      "3: CSV Logger\n"
                      "4: Model Checkpoint\n"
                      "5: Tensor Board\n"
                      "6: All\n")
                callback_choices = input(
                    "Choose the relevant number(s) separated"
                    " by a space: ")

                output_dict["callback_choices"] = [
                    int(size) for size in callback_choices.split()]

                options_list.remove(15)

            if 16 in options_list:
                print("\n16. Please choose which metrics you would like to "
                      "report\n"
                      "Options:\n"
                      "0: True Positives\n"
                      "1: False Positives\n"
                      "2: True Negatives\n"
                      "3: False Negatives\n"
                      "4: Accuracy\n"
                      "5: Precision\n"
                      "6: Recall\n"
                      "7: AUC (ROC Curve) \n"
                      "8: IOU\n"
                      "9: All")

                metric_choices = input(
                    "Choose the relevant number(s) separated by a space: ")

                output_dict["metric_choices"] = [
                    int(size) for size in metric_choices.split()]
                options_list.remove(16)

            if 17 in options_list:
                test_dir = input("\n17. If you would like to evaluate a test"
                                 " set please provide test directory. To"
                                 " skip this step leave this answer"
                                 " blank.\nNote, if providing a directory, "
                                 "please include a / at the end : ")

                output_dict["test_dir"] = test_dir

                options_list.remove(17)

            if 18 in options_list:
                filters = input(
                    "\n18. Please enter the filter multiple (filter multiple "
                    "is 2^multiple): ")

                output_dict["filters"] = int(filters)

                options_list.remove(18)

            if 19 in options_list:
                loss_weight = input(
                    "\n19. Please enter a loss weight (leave this blank to "
                    "skip, or used balanced weighting): ")
                if loss_weight:
                    output_dict["loss_weight"] = float(loss_weight)
                else:
                    output_dict["loss_weight"] = None

                options_list.remove(19)

            if 20 in options_list:
                initializer = input(
                    "\n20. Please choose the intializer you want to use:"
                    "Options:\n"
                    "0: He Normal\n"
                    "1: Glorot Uniform\n"
                    "Please choose the relevant number: "
                )

                if initializer == 0:
                    output_dict["initializer"] = "he_normal"
                else:
                    output_dict["initializer"] = "glorot_uniform"

                options_list.remove(20)

            if 21 in options_list:
                activation = input(
                    "\n21. Please choose the activation you want to use:"
                    "Options:\n"
                    "0: ReLU\n"
                    "1: SELU\n"
                    "Please choose the relevant number: "
                )

                if activation == 0:
                    output_dict["activation"] = "relu"
                else:
                    output_dict["activation"] = "selu"

                options_list.remove(21)

            if 22 in options_list:
                threshold = input(
                    "\n22. Please enter the threshold to use for predictions "
                    "(leave blank to use 0.5): "
                )

                if threshold:
                    output_dict["threshold"] = float(threshold)
                else:
                    output_dict["threshold"] = 0.5

                options_list.remove(22)

            if 23 in options_list:
                run_name = input(
                    "\n23. Please enter the run name, this will be"
                    " the name used to save your callback output"
                    " (if applicable): ")

                output_dict["run_name"] = run_name

                options_list.remove(23)

        print_options_dict(output_dict)

        options_list = input(
            "\nIf you are satisfied with this configurations, please enter 0."
            "\nIf not, please enter the number(s) of the options you would"
            " like to change.\nTo restart please enter -1."
            "\nSeparate multiple options by a space: ")
        options_list = set([int(choice) for choice in options_list.split()])

        if len(options_list) == 1 and 0 in options_list:
            happy = True
        elif 0 in options_list:
            print("\nYou entered options in addition to 0, so 0 will be "
                  "removed")
            options_list.remove(0)

    save_path = input("\nIf you would like to save this configuration"
                      " please enter the full json file name, including"
                      " the file path: ")

    if save_path:
        with open(save_path, 'w') as file:
            json.dump(output_dict, file)

    return output_dict
# *===========================================================================*
