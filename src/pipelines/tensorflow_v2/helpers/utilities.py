import argparse
from glob import glob
import json

import cv2
import numpy as np
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
        mask = tf.reshape(mask, mask_shape)
        # mask = tf.image.resize(mask, [512, 512])
        mask = tfa.image.gaussian_filter2d(mask)

        # will return 255 where the condition is met else 0
        mask = tf.where(tf.greater(mask, 0), 255, 0)
        mask = tf.cast(mask, tf.float32) / 255

        return img, mask
    return parse_image


# *=============================== load model ================================*
def check_model_save(old_pred, new_pred, old_bloss=None, new_bloss=None):
    np.testing.assert_allclose(old_pred, new_pred, atol=1e-6,
                               err_msg="Prediction from the saved model is"
                                       " not the same as the original model")

    if old_bloss:
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
          f"16. {'Run name':<40}: {answers['run_name']}\n")


def interactive_prompt():
    happy = False
    # list options = 1 - 16
    options_list = set(range(1, 17))

    print("Hello and welcome to plant-network-segmentation's training module"
          "\nA few inputs will be required from you to begin...")

    while not happy:
        if 1 in options_list:
            train_base_dir = input("\n1. Where is the base directory of your"
                                   " training images?\nNote, please include a"
                                   " / at the end of the directory: ")

            options_list.remove(1)

        if 2 in options_list:
            val_base_dir = input("\n2. Where is the base directory of your"
                                 " validation images?\nNote, please include a"
                                 " / at the end of the directory: ")

            options_list.remove(2)

        if 3 in options_list:
            leaf_ext, mask_ext = input(
                "\n3. What are the the leaf and mask extensions respectively?"
                "\nPlease provide the leaf extension first and separate your"
                " answers with a space\nNote, not . should be included in the"
                " ext name, e.g. \"png\": ").split()

            options_list.remove(3)

        if 4 in options_list:
            print("\n4. Should augmented images be included in the training "
                  "set?\n"
                  "Options:\n"
                  "0: False\n"
                  "1: True")
            incl_aug = input("Please choose a number: ")
            incl_aug = int(incl_aug) == 1

            options_list.remove(4)

        if 5 in options_list:
            leaf_shape = input("\n5. Please enter the leaf image shape, "
                               "separating"
                               " each number by a space: ")

            options_list.remove(5)

        if 6 in options_list:
            mask_shape = input("\n6. Please enter the mask image shape, "
                               "separating"
                               " each number by a space: ")

            options_list.remove(6)

        if 7 in options_list:
            batch_size = int(input("\n7. Please provide a batch size: "))

            options_list.remove(7)

        if 8 in options_list:
            buffer_size = int(input(
                "\n8. Please provide a buffer size\nNote, this influences the"
                " extent to which the data is shuffled: "))

            options_list.remove(8)

        if 9 in options_list:
            print("\n9. Please choose which model you would like to use\n"
                  "Options:\n"
                  "0: Vanilla U-Net\n"
                  "1: U-Net with ResNet backbone \n"
                  "2: W-Net\n")
            model_choice = int(input("Please choose the relevant model"
                                     " number: "))

            options_list.remove(9)

        if 10 in options_list:
            print("\n10. Please choose which loss function you would like to "
                  "use\n"
                  "Options:\n"
                  "0: Balance cross-entropy\n"
                  "1: Weighted cross-entropy \n"
                  "2: Focal loss\n"
                  "3: Soft dice loss\n")
            loss_choice = int(input("\nPlease choose the relevant loss"
                                    " function number: "))

            options_list.remove(10)

        if 11 in options_list:
            print("\n11. Please choose which optimiser you would like to use\n"
                  "Options:\n"
                  "0: Adam\n"
                  "1: SGD with momentum \n")
            opt_choice = int(input("Please choose the relevant optimiser"
                                   " number: "))

            options_list.remove(11)

        if 12 in options_list:
            lr = float(input("\n12. Please provide a learning rate: "))

            options_list.remove(12)

        if 13 in options_list:
            epochs = float(input("\n13. Please provide the number of epochs"
                                 " to run for: "))

            options_list.remove(13)

        if 14 in options_list:
            print("\n13. Please choose which callbacks you would like to use\n"
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

            options_list.remove(14)

        if 15 in options_list:
            print("\n14. Please choose which metrics you would like to "
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
                  "8: All")

            metric_choices = input("Choose the relevant number(s) separated by"
                                   " a space: ")

            options_list.remove(15)

        if 16 in options_list:
            run_name = input("\n15. Please enter the run name, this will be"
                             " the name used to save your callback output"
                             " (if applicable): ")

            options_list.remove(16)

        answers = {"train_base_dir": train_base_dir, "val_base_dir": val_base_dir,
                   "leaf_ext": leaf_ext, "mask_ext": mask_ext,
                   "incl_aug": incl_aug, "mask_shape": mask_shape,
                   "leaf_shape": leaf_shape, "batch_size": batch_size,
                   "buffer_size": buffer_size, "model_choice": model_choice,
                   "loss_choice": loss_choice, "opt_choice": opt_choice,
                   "lr": lr, "epochs": epochs, "run_name": run_name,
                   "callback_choices": callback_choices,
                   "metric_choices": metric_choices}

        print_user_input(answers)

        options_list = input(
            "\nIf you are satisfied with this configurations, please enter 0."
            "\nIf not, please enter the number(s) of the options you would"
            " like to change. Separate multiple options by a space: ")
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
            json.dump(answers, file)

    return answers


def format_input(answers_dict):
    to_tuple_keys = ["leaf_shape", "mask_shape"]

    for key in to_tuple_keys:
        answers_dict[key] = tuple(
            [int(size) for size in answers_dict[key].split()])

    to_list_keys = ["metric_choices", "callback_choices"]

    for key in to_list_keys:
        answers_dict[key] = [int(size) for size in answers_dict[key].split()]

    to_int_keys = ["batch_size", "buffer_size", "model_choice",
                   "loss_choice", "opt_choice", "epochs"]

    for key in to_int_keys:
        answers_dict[key] = int(answers_dict[key])

    answers_dict["lr"] = float(answers_dict["lr"])

    return answers_dict
# *===========================================================================*
