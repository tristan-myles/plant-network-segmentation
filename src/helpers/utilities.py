import argparse

import matplotlib as mpl

from src.data.data_model import *

from sklearn import metrics

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

    parser.add_argument("json_path", type=str,
                        help="path to a JSON file with the required "
                             "parameters to create LeafSequence and "
                             "MaskSequence objects")

    subparsers = parser.add_subparsers(title="actions",
                                       description='possible actions using '
                                                   'this module')

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
# *===========================================================================*
