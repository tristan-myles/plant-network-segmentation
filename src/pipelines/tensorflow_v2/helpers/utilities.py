import argparse
import json
from glob import glob
import pprint
from pathlib import Path
import logging
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve

from kerastuner import Objective
from kerastuner.tuners import BayesianOptimization

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# *================================= general =================================*
def save_prcurve_csv(run_name, mask, pred, type):
    y_true = np.array(mask).flatten()
    y_pred = np.array(pred).flatten()

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_json = {"precision": list(precision.astype("str")),
               "recall": list(recall.astype("str")),
               "thresholds": list(thresholds.astype("str"))}

    # Output path is handled differently compared to rest of file
    project_root = Path(os.getcwd())
    output_path = output_path = project_root.joinpath("data", "run_data",
                                                      "pr_curves")
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path.joinpath(run_name+"_" + type + "_pr_curve.json")

    with open(filename, "w") as file:
        json.dump(pr_json, file)


def save_predictions(prediction, folder_path, filename):
    output_folder_path = os.path.join(folder_path, "../predictions")
    folderpath = filename.split("/", 5)[5]
    folderpath = folderpath.rsplit("/", 2)[0]
    output_folder_path = os.path.join(output_folder_path, folderpath)
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    filename=filename.rsplit("/", 1)[1]
    filepath = os.path.join(output_folder_path, filename)

    cv2.imwrite(filepath, prediction)


def save_compilation_dict(answers, lr, save_path):
    model_choices = ["unet", "unet_resnet", "wnet"]
    loss_choices = ["bce", "wce", "focal", "dice"]
    opt_choices = ["adam", "sgd"]

    compilation_dict = {"model": model_choices[answers["model_choice"]],
                        "loss": loss_choices[answers["loss_choice"]],
                        "opt": opt_choices[answers["opt_choice"]],
                        "filters": answers["filters"],
                        "loss_weight": answers["loss_weight"],
                        "initializer":  answers["initializer"],
                        "activation": answers["activation"],
                        "lr": str(lr)}

    save_path = save_path + "compilation.json"
    with open(save_path, 'w') as file:
        json.dump(compilation_dict, file)


def get_sorted_list(search_string):
    return sorted([str(f) for f in glob(search_string, recursive=True)])


def configure_for_performance(ds, batch_size=2, buffer_size=100):
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


def tune_model(hyperModel, train_dataset, val_dataset, results_dir, run_name,
               input_shape, output_channels=1):
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


def save_lrt_results(lr_range_test, save_path):
    lrt_dict = {"smooth loss": lr_range_test.smoothed_losses,
                "batch num": lr_range_test.batch_nums,
                "log lr": lr_range_test.batch_log_lr}
    save_path = save_path + "_lrt_results.json"
    with open(save_path, 'w') as file:
        json.dump(lrt_dict, file)


def get_class_weight(training_path, incl_aug):
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
def global_contrast_normalization(img, s=1, lamb=10, eps=10e-8):
    total_mean = tf.math.reduce_mean(img)
    contrast = tf.math.reduce_mean(tf.math.pow(img - total_mean, 2))
    biased_contrast = tf.math.sqrt(lamb + contrast)

    denom = tf.math.maximum(biased_contrast, eps)

    normalized_image = tf.math.scalar_mul(s, (
        tf.math.divide((img - total_mean), denom)))

    return normalized_image


# *------------------------------ image parsing ------------------------------*
def read_file(img_path, shift_256, transform_uint8):
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


def parse_numpy_image(img, batch_shape):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, batch_shape)

    return img


def parse_image_fc(leaf_shape, mask_shape, test=False, shift_256=False,
                   transform_uint8=False):
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
def im2_lt_im1(pred, input_image):
    # reducing false positives, i.e trying to improve precision
    pred[(input_image >= 0)] = 0

    return pred


# *=============================== load model ================================*
def check_model_save(model, new_model, new_loss, new_opt, answers, metrics,
                     model_save_path, check_opt=False):
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


def get_model_pred_batch_loss(model, leaf_shape, mask_shape):
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
def print_user_input(answers):
    print(f"\nYour chosen configuration is:\n"
          f"1.  {'Training base directory':<40}: {answers['train_base_dir']}\n"
          f"    {'Shift 256':<40}: {answers['shift_256']}\n"
          f"    {'Transform uint8':<40}: {answers['transform_uint8']}\n"
          f"2.  {'Validation base directory':<40}: {answers['val_base_dir']}\n"
          f"3.  {'Leaf, Mask extension':<40}: {answers['leaf_ext']},"
          f" {answers['mask_ext']}\n"
          f"4.  {'Include augmented images':<40}: {answers['incl_aug']}\n"
          f"5.  {'Leaf shape':<40}: {answers['leaf_shape']}\n"
          f"6.  {'Mask shape':<40}: {answers['mask_shape']}\n"
          f"7.  {'Batch size':<40}: {answers['batch_size']}\n"
          f"8.  {'Buffer size':<40}: {answers['buffer_size']}\n"
          f"9.  {'Model choice':<40}: {answers['model_choice']}\n"
          f"10. {'Loss function choice':<40}: {answers['loss_choice']}\n"
          f"11. {'Optimiser choice':<40}: {answers['opt_choice']}\n"
          f"12. {'Learning rate':<40}: {answers['lr']}\n"
          f"13. {'Epochs':<40}: {answers['epochs']}\n"
          f"14. {'Callback choices':<40}: {answers['callback_choices']}\n"
          f"15. {'Metric choices':<40}: {answers['metric_choices']}\n"
          f"16. {'Run name':<40}: {answers['run_name']}\n"
          f"17. {'Test directory':<40}: {answers['test_dir']}\n"
          f"18. {'Filter multiple':<40}: {answers['filters']}\n"
          f"19. {'Loss weight':<40}: {answers['loss_weight']}\n",
          f"20. {'Initializer':<40}: {answers['initializer']}\n",
          f"21. {'Activation':<40}: {answers['activation']}\n")


def print_options_dict(output_dict):
    print(f"\nYour chosen configuration is:\n")
    for i, (key, val) in enumerate(output_dict.items()):
        if i == 0:
            num = "  "
        elif i == 1:
            num = str(i) + "."
        elif i == 2 or i == 3:
            num = "  "
        else:
            num = str(i-2) + "."

        if isinstance(val, dict):
            print_str = "\n" + pprint.pformat(val) + "\n"
        else:
            print_str = val

        print(f"{num} {(' '.join(key.split('_'))).capitalize() :<40}:"
              f" {print_str}")


def interactive_prompt():
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
            output_dict["which"] = ["tuning", "training"][operation-1]

            # list options = 1 - 18
            options_list.update(range(1, 23))

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
                    "\n4. Please provide the number of epochs to run for: "))

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
                callback_choices = input("Choose the relevant number(s) separated"
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
                )

                if activation == 0:
                    output_dict["activation"] = "relu"
                else:
                    output_dict["activation"] = "selu"

                options_list.remove(21)

            if 22 in options_list:
                run_name = input(
                    "\n22. Please enter the run name, this will be"
                    " the name used to save your callback output"
                    " (if applicable): ")

                output_dict["run_name"] = run_name

                options_list.remove(22)

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
