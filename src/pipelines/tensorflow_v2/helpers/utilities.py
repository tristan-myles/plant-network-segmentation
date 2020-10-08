import argparse
from glob import glob

import cv2
import tensorflow as tf
import tensorflow_addons as tfa


# *================================= general =================================*
def get_sorted_list(search_string):
    return sorted([str(f) for f in glob(search_string, recursive=True)])


def configure_for_performance(ds, batch_size=2, buffer_size=100):
    # ds.cache() speeds up performance but uses too much memory in this project
    # ds = ds.cache()
    # Number of batches to use when shuffling (calls x examples sequentially
    # => bigger buffer = better shuffle)
    ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size)

    # number of batches to prefetch
    ds = ds.prefetch(buffer_size=2)
    return ds


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
def read_file(img_path):
    img = tf.convert_to_tensor(
        cv2.imread(img_path.decode("utf-8"), cv2.IMREAD_UNCHANGED),
        dtype=tf.float32)

    return img


def parse_image_fc(leaf_shape, mask_shape):
    def parse_image(img_path: str, mask_path: str):
        # load the raw data from the file as a string
        img = tf.numpy_function(read_file, [img_path], tf.float32)
        img = tf.where(tf.greater_equal(img, tf.cast(0, "float32")),
                       tf.cast(0, "float32"), img)
        img = tf.reshape(img, leaf_shape)

        # Applying median filter
        # img = tfa.image.median_filter2d(img)
        img = tf.cast(img, tf.float32) / 255.0
        # img = global_contrast_normalization(img)

        # Masks
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(mask, tf.float32) / 255
        mask = tf.reshape(mask, mask_shape)
        # mask = tf.image.resize(mask, [512, 512])
        mask = tfa.image.gaussian_filter2d(mask)

        # will return 255 where the condition is met else 0
        mask = tf.where(tf.greater(mask, 0), 255, 0)
        # if mask is not binary then can use the code below to cast 255 to 1:
        return img, mask
    return parse_image


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
def interactive_prompt():
    print("Hello and welcome to plant-network-segmentation's training module"
          "\nA few inputs will be required from you to begin...")
    train_base_dir = input("\nWhere is the base directory of your training "
                           "images?\nNote, please include a / at the end of "
                           "the directory: ")
    val_base_dir = input("\nWhere is the base directory of your validation "
                         "images?\nNote, please include a / at the end of "
                         "the directory: ")
    leaf_ext, mask_ext = input(
        "\nWhat are the the leaf and mask extensions respectively?"
        "\nPlease provide the leaf extension first and separate your answers "
        "with a space\nNote, not . should be included in the ext name, "
        "e.g. \"png\": ").split()

    print("\nShould augmented images be included in the training set?\n"
          "Options:\n"
          "0: True\n"
          "1: False")
    incl_aug = input("Please choose a number: ")
    incl_aug = int(incl_aug) == 0

    leaf_shape = tuple([int(size) for size in input(
        "Please enter the leaf image shape, separating each number by a "
        "space: ").split()])

    mask_shape = tuple([int(size) for size in input(
        "Please enter the mask image shape, separating each number by a "
        "space: ").split()])

    batch_size = int(input("\nPlease provide a batch size: "))

    buffer_size = int(input(
        "\nPlease provide a buffer size\nNote, this influences the extent to "
        "which the data is shuffled: "))

    print("\nPlease choose which model you would like to use\n"
          "Options:\n"
          "0: Vanilla U-Net\n"
          "1: U-Net with ResNet backbone \n"
          "2: W-Net\n")
    model_choice = int(input("Please choose the relevant model number: "))

    print("\nPlease choose which loss function you would like to use\n"
          "Options:\n"
          "0: Balance cross-entropy\n"
          "1: Weighted cross-entropy \n"
          "2: Focal loss\n"
          "3: Soft dice loss\n")
    loss_choice = int(input("\nPlease choose the relevant loss function "
                            "number: "))

    print("\nPlease choose which optimiser you would like to use\n"
          "Options:\n"
          "0: Adam\n"
          "1: SGD with momentum \n")
    opt_choice = int(input("Please choose the relevant optimiser number: "))

    lr = float(input("\nPlease provide a learning rate: "))

    print("\nPlease choose which callbacks you would like to use\n"
          "Options:\n"
          "0: Learning rate range test\n"
          "1: 1cycle policy\n"
          "2: Early stopping\n"
          "3: CSV Logger\n"
          "4: Model Checkpoint\n"
          "5: Tensor Board\n"
          "6: All\n")
    callback_choices = [int(choice) for choice in input(
        "Choose the relevant number(s) separated by a space: ").split()]

    print("\nPlease choose which metrics you would like to report\n"
          "Options:\n"
          "0: True Positives\n"
          "1: False Positives\n"
          "2: True Negatives\n"
          "3: False Negatives\n"
          "4: Accuracy\n"
          "5: Precision\n"
          "6: Recall\n"
          "7: AUC (ROC Curve) \n"
          "8: All")
    metric_choices = [int(choice) for choice in input(
        "Choose the relevant number(s) separated by a space: ").split()]

    print(f"\nYour chosen configuration is:\n"
          f"{'Training base directory':<40}: {train_base_dir}\n"
          f"{'Validation base directory':<40}: {val_base_dir}\n"
          f"{'Leaf, Mask extension':<40}: {leaf_ext}, {mask_ext}\n"
          f"{'Include augmented images':<40}: {incl_aug}\n"
          f"{'Leaf shape':<40}: {leaf_shape}\n"
          f"{'Mask shape':<40}: {mask_shape}\n"
          f"{'Batch size':<40}: {batch_size}\n"
          f"{'Buffer size':<40}: {buffer_size}\n"
          f"{'Model choice':<40}: {model_choice}\n"
          f"{'Loss function choice':<40}: {loss_choice}\n"
          f"{'Optimiser choice':<40}: {opt_choice}\n"
          f"{'Learning rate':<40}: {lr}\n"
          f"{'Callback choices':<40}: {callback_choices}\n"
          f"{'Metric choices':<40}: {metric_choices}\n")

    answers = {"train_base_dir": train_base_dir, "val_base_dir": val_base_dir,
               "leaf_ext": leaf_ext, "mask_ext": mask_ext,
               "incl_aug": incl_aug, "mask_shape": mask_shape,
               "leaf_shape": leaf_shape, "batch_size": batch_size,
               "buffer_size": buffer_size, "model_choice": model_choice,
               "loss_choice": loss_choice, "opt_choice": opt_choice,
               "lr": lr, "callback_choices": callback_choices,
               "metric_choices": metric_choices}

    return answers
# *===========================================================================*
