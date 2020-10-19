import argparse
import json
import pprint

import matplotlib as mpl
from sklearn import metrics

from src.data.data_model import *

LOGGER = logging.getLogger(__name__)


# *================================= metrics =================================*
def get_iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def classification_report(predictions, masks, save_path=None):
    metric_df = pd.DataFrame({"IoU": [], "Precision": [], "Recall": [],
                              "F1": [], "Accuracy": [], "AUC_PR": [],
                              "FN": [], "FP": [], "TN": [], "TP": []})

    for pred, mask in zip(predictions, masks):
        # grab the original image and reconstructed image
        y_true = mask.round().flatten()
        y_pred = pred.flatten()

        avg_pr = metrics.average_precision_score(y_true, y_pred)

        y_pred = y_pred.round()

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
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

        if save_path:
            metric_df.to_csv(save_path)

    return metric_df


# *=============================== data model ================================*
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


# *================================== plots ==================================*
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


# *============================== all __main__ ===============================*
def create_sequence_objects(sequence_input):
    lseqs = []
    mseqs = []

    leaf_dict = sequence_input["leaves"]["input"]
    if leaf_dict is not None:
        leaf_keys = list(leaf_dict.keys())
        for vals in zip(*list(leaf_dict.values())):
            kwargs = {key: val for key, val in zip(leaf_keys, vals)}
            lseqs.append(LeafSequence(**kwargs))

    mask_dict = sequence_input["masks"]["input"]
    if mask_dict is not None:
        mask_keys = list(mask_dict.keys())
        for vals in zip(*list(mask_dict.values())):
            kwargs = {key: val for key, val in zip(mask_keys, vals)}
            mseqs.append(MaskSequence(**kwargs))

    return lseqs, mseqs


def load_image_objects(seq_objects, load_images=False):
    LOGGER.info(f"Creating image objects for "
                f"{seq_objects[0].__class__.__name__}")

    for seq in seq_objects:
        LOGGER.info(f"Creating {seq.num_files} objects")
        seq.load_extracted_images(load_image=load_images)


# *============================ package __main__ =============================*
# *------------------------------- procedures --------------------------------*
def trim_sequence_images(seq_objects, x_size_dir_list=None,
                         y_size_dir_list=None, overwrite=False):
    for seq, x_size_dir, y_size_dir in \
            zip(seq_objects, x_size_dir_list, y_size_dir_list):
        seq.load_image_array()

        seq.trim_image_sequence(x_size_dir, y_size_dir, overwrite)

        seq.unload_extracted_images()


# *-------------------------------- argparse ---------------------------------*
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Perform operations using the "
                                     "plant-image-segmentation code base")

    parser.add_argument(
        "-i", "--interactive", action="store_true", default=False,
        help="flag to run the script in interactive mode")

    parser.add_argument("-j", "--json_path", metavar="\b", type=str,
                        help="path to input parameters for an action")

    subparsers = parser.add_subparsers(
        title="actions", description='possible actions using this module')

    parser.add_argument("-fj", "--filepath_json", metavar="\b", type=str,
                        help="path to a JSON file with the required "
                             "parameters to create LeafSequence and "
                             "MaskSequence objects")

    parser_extract_images = subparsers.add_parser("extract_images",
                                                  help="extraction help")
    parser_extract_images.set_defaults(which='extract_images')

    parser_extract_images.add_argument(
        "--leaf_output_path", "-lo", metavar="\b",
        help="output paths, if the paths are in the input json enter "
             "\"same\"")

    parser_extract_images.add_argument(
        "--mask_output_path", "-mo", metavar="\b",
        help="output paths, if the paths are in the input json enter "
             "\"same\"")

    parser_extract_images.add_argument(
        "--overwrite", "-o", action="store_true", default=False,
        help="overwrite existing images, note this flag is applied to both "
             "mask and leaf images")

    parser_extract_images.add_argument(
        "--binarise", "-b", action="store_true", default=False,
        help="save binary masks")

    parser_extract_tiles = subparsers.add_parser("extract_tiles",
                                                 help="extraction help")
    parser_extract_tiles.set_defaults(which="extract_tiles")

    parser_extract_tiles.add_argument("-sx", "--stride_x", metavar="\b",
                                      type=int, help="x stride size")
    parser_extract_tiles.add_argument("-sy", "--stride_y", metavar="\b",
                                      type=int, help="y stride size")
    parser_extract_tiles.add_argument("-lx", "--length_x", metavar="\b",
                                      type=int, help="tile x length")
    parser_extract_tiles.add_argument("-ly", "--length_y", metavar="\b",
                                      type=int, help="tile y length")

    parser_extract_tiles.add_argument(
        "--leaf_output_path", "-lo", metavar="\b",
        help="output paths, if you want to use "
             "the default path enter  \"default\", if the paths are in "
             "the input json enter  \"same\"")

    parser_extract_tiles.add_argument(
        "--mask_output_path", "-mo", metavar="\b",
        help="output paths, if you want to use "
             "the default path enter  \"default\", if the paths are in "
             "the input json enter  \"same\"")

    parser_trim_sequence = subparsers.add_parser(
        "trim_sequence", help="trims every image in an image sequence ")
    parser_trim_sequence.set_defaults(which='trim_sequence')

    parser_trim_sequence.add_argument(
        "--mask", "-m", action="store_true", default=False,
        help="whether the mask sequence should be trimmed, default is for "
             "the leaf sequence to be trimmed")
    parser_trim_sequence.add_argument(
        "--y_size_dir", "-ysd", metavar="\b",
        help="y output size and direction to be passed in as a tuple, "
             "where a 1 or -1 indicated to trim either top or bottom "
             "respectively")
    parser_trim_sequence.add_argument(
        "--x_size_dir", "-xsd", metavar="\b",
        help="x output size and direction to be passed in as a tuple, "
             "where a 1 or -1 indicated to trim either left or right "
             "respectively")
    parser_trim_sequence.add_argument(
        "--overwrite", "-o", action="store_true", default=False,
        help="whether or not the image being trimmed should be overwritten")

    parser_plot_profile = subparsers.add_parser(
        "plot_profile", help="plot an embolism profile")
    parser_plot_profile.set_defaults(which="plot_profile")
    parser_plot_profile.add_argument(
        "--output_path", "-o", type=str, metavar="\b", help="The plot output "
                                                            "path")
    parser_plot_profile.add_argument(
        "--show", "-s", action="store_true", help="flag indicating if the "
                                                  "plot should be shown")
    parser_plot_profile.add_argument(
        "--leaf_names", "-ln", type=str, metavar="\b",
        help="leaf names to be used in plot title")

    parser_plot_embolism_counts = subparsers.add_parser(
        "plot_embolism_counts", help="plot the embolism count profile for a "
                                     "dataset")
    parser_plot_embolism_counts.set_defaults(which="plot_embolism_counts")
    parser_plot_embolism_counts.add_argument(
        "--output_path", "-o", type=str, metavar="\b", help="The plot output "
                                                            "path")
    parser_plot_embolism_counts.add_argument(
        "--show", "-s", action="store_true", help="flag indicating if the "
                                                  "plot should be shown")
    parser_plot_embolism_counts.add_argument(
        "--leaf_names", "-ln", type=str, metavar="\b",
        help="leaf names to be used in plot title")
    parser_plot_embolism_counts.add_argument(
        "--tile", "-t", action="store_true",
        help="indicates if the plot should be created using tiles")
    parser_plot_embolism_counts.add_argument(
        "--leaf_embolism_only", "-leo", action="store_true",
        help="should only full leaves with embolisms be used")
    parser_plot_embolism_counts.add_argument(
        "--percent", "-p", action="store_true",
        help="should the plot y-axis be expressed as a percent")

    parser_eda_df = subparsers.add_parser(
        "eda_df", help="extract an eda dataframe")
    parser_eda_df.set_defaults(which="eda_df")

    parser_eda_df.add_argument(
        "csv_output_path", help="output paths, if the paths are in the input "
                                "json enter \"same\"")
    parser_eda_df.add_argument("--tiles", "-t", action="store_true")

    parser_databunch_df = subparsers.add_parser(
        "databunch_df", help="extract an databunch dataframe")
    parser_databunch_df.set_defaults(which="databunch_df")

    parser_databunch_df.add_argument(
        "csv_output_path", help="output paths, if the paths are in the input"
                                " json enter \"same\"")

    parser_databunch_df.add_argument("--tiles", "-t", action="store_true")
    parser_databunch_df.add_argument("--tile_embolism_only", "-teo",
                                     action="store_true",
                                     help="should only tiles with embolisms "
                                          "be used")
    parser_databunch_df.add_argument(
        "--leaf_embolism_only", "-leo", action="store_true",
        help="should only full leafs with embolisms be used")

    args = parser.parse_args()
    return args


# *--------------------------- interactive prompt ----------------------------*
def print_options_dict(output_dict):
    print(f"\nYour chosen configuration is:\n")
    for i, (key, val) in enumerate(output_dict.items()):
        if i == 0:
            num = "  "
        else:
            num = str(i) + "."
        if isinstance(val, dict):
            print_str = "\n" + pprint.pformat(val) + "\n"
        else:
            print_str = val

        print(f"{num} {(' '.join(key.split('_'))).capitalize() :<20}:"
              f" {print_str}")


def interactive_prompt():
    happy = False
    options_list = set((-1,))
    operation_names = ["extract_images", "extract_tiles", "plot_profile",
                       "plot_embolism_counts", "eda_df", "databunch_df",
                       "trim_sequence"]

    while not happy:
        if -1 in options_list:
            output_dict = {}

            print("* What action would you like to take?\n"
                  "------------ Extraction ------------\n"
                  "1. Extract images\n"
                  "2. Extract tiles\n"
                  "------------- Plotting -------------\n"
                  "3. Plot embolism profile\n"
                  "4. Plot embolism count barplot\n"
                  "--------------- EDA ----------------\n"
                  "5. EDA DataFrame\n"
                  "6. DataBunch DataFrame\n"
                  "------------- General --------------\n"
                  "7. Trim sequence")

            operation = int(input("Please select your option: "))
            output_dict["which"] = operation_names[operation-1]

            # Include the max of all options: 1 - 10
            options_list.update(range(1, 10))

            options_list.remove(-1)
        print("Please separate multiple answers by a space")

        if 1 in options_list:
            leaf_input_path = input(
                "\n1. Where are the leaf images that you would like to"
                " use?\nPlease include the filename pattern if"
                " necessary\n(Leave this blank to skip)\nAnswer: ")

            output_dict["leaves"] = {
                "input": {"folder_path": [], "filename_pattern": []}}

            for path in leaf_input_path.split():
                folder_path, filename = path.rsplit("/", 1)
                output_dict["leaves"]["input"]["folder_path"].append(
                    folder_path + "/")
                output_dict["leaves"]["input"]["filename_pattern"].append(
                    filename)

            options_list.remove(1)

        if 2 in options_list:
            mask_input_path = input(
                "\n2. Where are the mask images that you would like to"
                " use include the filename pattern if"
                " necessary\n(Leave this blank to skip)\nAnswer: ")

            if operation == 1:
                output_dict["masks"] = {
                    "input": {"mpf_path": mask_input_path.split()}}
            else:
                output_dict["masks"] = {
                    "input": {"folder_path": [], "filename_pattern": []}}

                for path in mask_input_path.split():
                    folder_path, filename = path.rsplit("/", 1)
                    output_dict["masks"]["input"]["folder_path"].append(
                        folder_path+"/")
                    output_dict["masks"]["input"]["filename_pattern"].append(
                        filename)

                options_list.remove(2)

        if operation == 1 or operation == 2:
            if 3 in options_list:
                if leaf_input_path:
                    leaf_output_path = input(
                        "\n3. Where do you want to save the extracted leaves."
                        "\n   Please also include the file name."
                        "\nAnswer: ")

                    output_dict["leaf_output_path"] = leaf_output_path
                else:
                    print("\nYou did not enter a leaf directory, so question 3"
                          " will be skipped.")
                    output_dict["leaf_output_path"] = None

                options_list.remove(3)

            if 4 in options_list:
                if mask_input_path:
                    mask_output_path = input(
                        "\n4. Where do you want to save the extracted masks."
                        "\n   Please also include the file name."
                        "\nAnswer: ")

                    output_dict["mask_output_path"] = mask_output_path
                else:
                    print("\nYou did not enter a mask directory, so question 4"
                          " will be skipped.")

                    output_dict["mask_output_path"] = None

                options_list.remove(4)

            if 5 in options_list:
                print("\n5. Would you like to overwrite any existing "
                      "extracted images?\n"
                      "   Options:\n"
                      "   0: False\n"
                      "   1: True")
                overwrite = input("Please choose a number: ")
                overwrite = int(overwrite) == 1

                output_dict["overwrite"] = overwrite

                options_list.remove(5)

        if operation == 1:
            if 6 in options_list:
                print("\n6. Would you like to save binarised masks - i.e."
                      " pixel values as 0, 1?\n"
                      "   Options:\n"
                      "   0: False\n"
                      "   1: True")
                binarise = input("Please choose a number: ")
                binarise = int(binarise) == 1

                output_dict["binarise"] = binarise

                options_list.remove(6)

        if operation == 2:
            if 6 in options_list:
                sx = input("\n6. Please enter the size of the x stride: ")

                output_dict["stride_x"] = int(sx)

                options_list.remove(6)

            if 7 in options_list:
                sy = input("\n7. Please enter the size of the y stride: ")

                output_dict["stride_y"] = int(sy)

                options_list.remove(7)

            if 8 in options_list:
                lx = input("\n8. Please enter the width of the tile"
                           " (x length): ")

                output_dict["length_x"] = int(lx)

                options_list.remove(8)

            if 9 in options_list:
                ly = input("\n9. Please enter the length of the tile"
                           " (y length): ")

                output_dict["length_y"] = int(ly)

                options_list.remove(9)

        if operation == 3 or operation == 4:
            if 3 in options_list:
                output_path = input(
                    "\n3. Please enter the image output paths, enter a"
                    " path, including the filename, for each image."
                    "\nAnswer: ")

                output_dict["output_path"] = output_path

                options_list.remove(3)

            if 4 in options_list:
                leaf_names = input(
                    "\n4. What leaf names should be used in the plot "
                    "title?\nPlease enter a name per leaf: ")

                output_dict["leaf_names"] = leaf_names

                options_list.remove(4)

            if 5 in options_list:
                print("\n5. Would you like to display the plot before exiting"
                      " the script?\n"
                      "   Options:\n"
                      "   0: No\n"
                      "   1: Yes")
                show = input("Please choose a number: ")
                show = int(show) == 1

                output_dict["show"] = show

                options_list.remove(5)

        if operation == 4 or operation == 6:
            if 6 in options_list:
                print("\n6. Should only leaves with embolisms be used?\n"
                      "   Options:\n"
                      "   0: No\n"
                      "   1: Yes")
                leo = input("Please choose a number: ")
                leo = int(leo) == 1

                output_dict["leaf_embolism_only"] = leo

                options_list.remove(6)

        if operation == 4:
            if 7 in options_list:
                print("\n7. Would you like to create the plot using image "
                      "tiles\n"
                      "   Options:\n"
                      "   0: No\n"
                      "   1: Yes")
                tile = input("Please choose a number: ")
                tile = int(tile) == 1

                output_dict["tile"] = tile

                options_list.remove(7)

            if 8 in options_list:
                print("\n8. Should the y-axis scale use percentages?\n"
                      "   Options:\n"
                      "   0: No\n"
                      "   1: Yes")
                percent = input("Please choose a number: ")
                percent = int(percent) == 1

                output_dict["percent"] = percent

                options_list.remove(8)

        if operation == 5 or operation == 6:
            if 3 in options_list:
                csv_output_path = input(
                    "\n3. Where do you want to save the csv output?."
                    "\n   Please also include the file name."
                    "\nAnswer: ")

                output_dict["csv_output_path"] = csv_output_path

                options_list.remove(3)

            if 4 in options_list:
                print("\n4. Would you like to create the data frame using"
                      " tiles only\n"
                      "   Options:\n"
                      "   0: No\n"
                      "   1: Yes")
                tiles = input("Please choose a number: ")
                tiles = int(tiles) == 1

                output_dict["tiles"] = tiles

                options_list.remove(4)

            if 5 in options_list and operation == 5:
                print("\n5. Please choose which fields you would like included"
                      " in the EDA DataFrame\n"
                      "Options:\n"
                      "0: Linked filename\n"
                      "1: Unique range\n"
                      "2: Embolism percent\n"
                      "3: Intersection\n"
                      "4: Has embolism\n")

                eda_options = input("Choose the relevant number(s) separated"
                                    " by a space: ").split()

                output_dict["eda_df_options"] = {"linked_filename": False,
                                                 "unique_range": False,
                                                 "embolism_percent": False,
                                                 "intersection": False,
                                                 "has_embolism": False}

                eda_df_options = list(output_dict["eda_df_options"])

                for option in eda_options:
                    output_dict["eda_df_options"][eda_df_options[int(option)]] = \
                        True

            options_list.remove(5)

        if operation == 6:
            if 5 in options_list:
                print("\n5. Should only tiles with embolisms be used?\n"
                      "   Options:\n"
                      "   0: No\n"
                      "   1: Yes")
                teo = input("Please choose a number: ")
                teo = int(teo) == 1

                output_dict["tile_embolism_only"] = teo

        if operation == 7:
            if 3 in options_list:
                ysd = input(
                    "\n3. What is the y output size and directions?\n   Note,"
                    " for direction, a 1 or -1 indicates to trim either the"
                    " top or bottom respectively."
                    "\n Reminder, separate your answers by a space."
                    "\nAnswer: ")

                output_dict["y_size_dir"] = ysd

                options_list.remove(3)

            if 4 in options_list:
                xsd = input(
                    "\n4. What is the x output size and directions?\n   Note,"
                    " for direction, a 1 or -1 indicates to trim either the"
                    " left or right respectively."
                    "\n Reminder, separate your answers by a space."
                    "\nAnswer: ")

                output_dict["x_size_dir"] = xsd

                options_list.remove(4)

            if 5 in options_list:
                print("\n5. Would you like to overwrite any existing "
                      "extracted images?\n"
                      "   Options:\n"
                      "   0: False\n"
                      "   1: True")
                overwrite = input("Please choose a number: ")
                overwrite = int(overwrite) == 1

                output_dict["overwrite"] = overwrite

                options_list.remove(5)

            if 6 in options_list:
                print("\n6. Would you like to trim mask images (instead of "
                      "leaf images)?\n"
                      "   Options:\n"
                      "   0: No\n"
                      "   1: Yes")
                trim_masks = input("Please choose a number: ")
                trim_masks = int(trim_masks) == 1

                output_dict["mask"] = trim_masks

                options_list.remove(6)

        if operation not in list(range(1, 8)):
            raise ValueError("Please choose an option from the input list")

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
