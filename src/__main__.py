import argparse
import json
import logging.config
from ast import literal_eval
from os import path

from src.data.data_model import *
from src.eda.describe_leaf import plot_embolisms_per_leaf
from src.helpers.utilities import *

abs_path = path.dirname(path.abspath(__file__))

logging.config.fileConfig(fname=abs_path + "/logging_configuration.ini",
                          defaults={'logfilename': abs_path + "/main.log"},
                          disable_existing_loggers=False)
LOGGER = logging.getLogger(__name__)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


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


def extract_leaf_images(lseq_list, output_path_list, overwrite):
    LOGGER.info("Extracting differenced leaf images")
    for lseq, output_path in zip(lseq_list, output_path_list):
        LOGGER.info(f"Differencing images in {lseq.folder_path} and saving "
                    f"to {output_path}")
        lseq.extract_changed_leaves(output_path, overwrite=overwrite)


def extract_multipage_mask_images(mseq_list, output_path_list,
                                  overwrite, binarise):
    LOGGER.info("Extracting mask images from multipage file")

    for mseq, output_path in zip(mseq_list, output_path_list):
        LOGGER.info(f"Extracting images from: {mseq.mpf_path} and saving "
                    f"to {output_path}")
        mseq.extract_mask_from_multipage(output_path,
                                         overwrite,
                                         binarise)

        # frees up ram when extracting many sequences
        mseq.unload_extracted_images()


def extract_tiles(seq_objects, length_x, stride_x, length_y,
                  stride_y, output_path_list=None):
    LOGGER.info(f"Extracting tiles from {seq_objects[0].__class__.__name__} "
                f"with the following configuration:"
                f" length (x, y): ({length_x}, {length_y}) |"
                f" stride (x, y): ({stride_x}, {stride_y})")

    if output_path_list is None:
        for seq in seq_objects:
            seq.tile_sequence(length_x=length_x, stride_x=stride_x,
                              length_y=length_y, stride_y=stride_y)
    else:
        for seq, output_path in zip(seq_objects, output_path_list):
            seq.tile_sequence(length_x=length_x, stride_x=stride_x,
                              length_y=length_y, stride_y=stride_y,
                              output_path=output_path)


def trim_sequence_images(seq_objects, x_size_dir_list=None,
                         y_size_dir_list=None, overwrite=False):
    for seq, x_size_dir, y_size_dir in \
            zip(seq_objects, x_size_dir_list, y_size_dir_list):
        seq.load_image_array()

        seq.trim_image_sequence(x_size_dir, y_size_dir, overwrite)

        seq.unload_extracted_images()


def plot_mseq_profiles(mseqs, show, output_path_list, leaf_name_list):
    for i, mseq in enumerate(mseqs):
        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images(load_image=True)

        LOGGER.info("Extracting the intersection list")
        mseq.get_intersection_list()

        LOGGER.info("Extracting the embolism percent list")
        mseq.get_embolism_percent_list()

        if output_path_list is not None:
            mseq.plot_profile(show=show, output_path=output_path_list[i],
                              leaf_name=leaf_name_list[i], figsize=(10, 15))
        else:
            mseq.plot_profile(show=show, leaf_name=leaf_name_list[i],
                              figsize=(10, 15))

        mseq.unload_extracted_images()


def plot_mseq_embolism_counts(mseqs, show, output_path, tiles,
                              leaf_name_list, leaf_embolism_only=False,
                              percent=False):
    has_embolism_lol = []

    for i, mseq in enumerate(mseqs):
        has_embolism_lol.append([])

        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images()

        if not tiles or leaf_embolism_only:
            mseq.load_image_array()

            LOGGER.info("Extracting the embolism percent list")
            mseq.get_embolism_percent_list()

            LOGGER.info("Extracting the has_embolism list")
            mseq.get_has_embolism_list()

            mseq.unload_extracted_images()

        if not tiles:
            has_embolism_lol[i] = has_embolism_lol[i] + mseq.has_embolism_list

        if tiles:
            with tqdm(total=len(mseq.image_objects), file=sys.stdout) as pbar:
                for mask_image in mseq.image_objects:
                    if leaf_embolism_only:
                        # Assumes linked sequences have been provided.
                        if not mask_image.has_embolism:
                            # Skip this image if the mask has no embolism
                            continue

                    mask_image.load_tile_paths()
                    mask_image.load_extracted_images(load_image=True,
                                                     disable_pb=True)
                    mask_image.get_embolism_percent_list(disable_pb=True)
                    mask_image.get_has_embolism_list(disable_pb=True)
                    has_embolism_lol[i] = \
                        has_embolism_lol[i] + mask_image.has_embolism_list
                    mask_image.unload_extracted_images()
                    pbar.update(1)

    if output_path is not None:
        plot_embolisms_per_leaf(has_embolism_lol=has_embolism_lol,
                                show=show, output_path=output_path,
                                leaf_names_list=leaf_name_list,
                                percent=percent, figsize=(15, 10))
    else:
        plot_embolisms_per_leaf(has_embolism_lol=has_embolism_lol,
                                show=show, leaf_names_list=leaf_name_list,
                                percent=percent, figsize=(15, 10))

        mseq.unload_extracted_images()


def extract_full_eda_df(mseqs, options, output_path_list, lseqs=None):
    for i, (mseq, csv_output_path) in enumerate(zip(mseqs, output_path_list)):
        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images(load_image=True)

        if options["linked_filename"]:
            mseq.link_sequences(lseqs[i])

        _ = mseq.get_eda_dataframe(options, csv_output_path)

        mseq.unload_extracted_images()


def extract_tiles_eda_df(mseqs, options, output_path_list, lseqs=None):
    for i, (mseq, csv_output_path) in enumerate(zip(mseqs, output_path_list)):
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")

        if options["linked_filename"]:
            mseq.link_sequences(lseqs[i])

        _ = mseq.get_tile_eda_df(options, csv_output_path)

        mseq.unload_extracted_images()


def extract_full_databunch_df(lseqs, mseqs, output_path_list,
                              embolism_only=False):
    for csv_output_path, lseq, mseq in zip(output_path_list, lseqs, mseqs):
        mseq.load_extracted_images(load_image=True)
        mseq.get_embolism_percent_list()
        mseq.get_has_embolism_list()

        lseq.link_sequences(mseq)

        LOGGER.info(f"Creating DataBunch DataFrame using "
                    f"{lseq.__class__.__name__} located at {lseq.folder_path} "
                    f"and {mseq.__class__.__name__} located at"
                    f" {mseq.folder_path}")

        # get_databunch_dataframe written into of a lseq, i.e. will always
        # report lseq regardless of whether it is longer or shorter
        _ = lseq.get_databunch_dataframe(embolism_only, csv_output_path)


def extract_tiles_databunch_df(lseqs, mseqs, output_path_list,
                               tile_embolism_only=False,
                               leaf_embolism_only=False):
    for csv_output_path, lseq, mseq in zip(output_path_list, lseqs, mseqs):
        lseq.link_sequences(mseq)

        LOGGER.info(f"Creating Tile DataBunch DataFrame using "
                    f"{lseq.__class__.__name__} located at {lseq.folder_path} "
                    f"and {mseq.__class__.__name__} located at"
                    f" {mseq.folder_path}")

        _ = lseq.get_tile_databunch_df(mseq, tile_embolism_only,
                                       leaf_embolism_only,
                                       csv_output_path)


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
        help="should only full leafs with embolisms be used")
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


if __name__ == "__main__":
    ARGS = parse_arguments()

    LOGGER.debug(ARGS.which)
    with open(ARGS.json_path, "r") as JSON_FILE:
        INPUT_JSON_DICT = json.load(JSON_FILE)

    if ARGS.which != "train_fastai":
        LSEQS, MSEQS = create_sequence_objects(INPUT_JSON_DICT)

    if ARGS.which == "extract_images":
        if ARGS.leaf_output_path is not None:
            if ARGS.leaf_output_path == "same":
                LEAF_OUTPUT_LIST = \
                    INPUT_JSON_DICT["leaves"]["output"]["output_path"]
            else:
                LEAF_OUTPUT_LIST = str.split(ARGS.leaf_output_path, " ")

            extract_leaf_images(LSEQS, LEAF_OUTPUT_LIST, ARGS.overwrite)

        if ARGS.mask_output_path is not None:
            if ARGS.mask_output_path == "same":
                MASK_OUTPUT_LIST = \
                    INPUT_JSON_DICT["masks"]["output"]["output_path"]
            else:
                MASK_OUTPUT_LIST = str.split(ARGS.mask_output_path, " ")

            extract_multipage_mask_images(MSEQS, MASK_OUTPUT_LIST,
                                          ARGS.overwrite, ARGS.binarise)

    if ARGS.which == "extract_tiles":
        if ARGS.leaf_output_path is not None:
            load_image_objects(LSEQS)

            if ARGS.leaf_output_path == "default":
                LEAF_OUTPUT_LIST = None
            elif ARGS.leaf_output_path == "same":
                LEAF_OUTPUT_LIST = \
                    INPUT_JSON_DICT["leaves"]["output"]["output_path"]
            else:
                LEAF_OUTPUT_LIST = str.split(ARGS.leaf_output_path, " ")

            extract_tiles(LSEQS, ARGS.length_x, ARGS.stride_x,
                          ARGS.length_y, ARGS.stride_y, LEAF_OUTPUT_LIST)

        if ARGS.mask_output_path is not None:
            load_image_objects(MSEQS)

            if ARGS.mask_output_path == "default":
                MASK_OUTPUT_LIST = None
            elif ARGS.mask_output_path == "same":
                MASK_OUTPUT_LIST = \
                    INPUT_JSON_DICT["masks"]["output"]["output_path"]
            else:
                MASK_OUTPUT_LIST = str.split(ARGS.mask_output_path, " ")

            extract_tiles(MSEQS, ARGS.length_x, ARGS.stride_x,
                          ARGS.length_y, ARGS.stride_y, MASK_OUTPUT_LIST)

    if ARGS.which == "trim_sequence":
        if ARGS.mask:
            SEQS = MSEQS
        else:
            SEQS = LSEQS

        if ARGS.x_size_dir == "same":
            X_SIZE_DIR_LIST = INPUT_JSON_DICT["trim"]["x_size_dir"]
            X_SIZE_DIR_LIST = [tuple(X_SIZE_DIR) if X_SIZE_DIR else None for
                               X_SIZE_DIR in X_SIZE_DIR_LIST]
        elif not ARGS.x_size_dir:
            X_SIZE_DIR_LIST = [None]
        else:
            X_SIZE_DIR_LIST = literal_eval(ARGS.x_size_dir)

            # in the case of a single sequence with no x adjustment
            if (not X_SIZE_DIR_LIST or
                    isinstance(X_SIZE_DIR_LIST[0], int)):
                X_SIZE_DIR_LIST = [X_SIZE_DIR_LIST]

        if ARGS.y_size_dir == "same":
            Y_SIZE_DIR_LIST = INPUT_JSON_DICT["trim"]["y_size_dir"]
            Y_SIZE_DIR_LIST = [tuple(Y_SIZE_DIR) if Y_SIZE_DIR else None for
                               Y_SIZE_DIR in Y_SIZE_DIR_LIST]
        elif not ARGS.y_size_dir:
            Y_SIZE_DIR_LIST = [None]
        else:
            Y_SIZE_DIR_LIST = literal_eval(ARGS.y_size_dir)

            # in the case of a single sequence with no y adjustment
            if (not Y_SIZE_DIR_LIST or
                    isinstance(Y_SIZE_DIR_LIST[0], int)):
                Y_SIZE_DIR_LIST = [Y_SIZE_DIR_LIST]

        if ARGS.y_size_dir:
            load_image_objects(SEQS)
            trim_sequence_images(SEQS, X_SIZE_DIR_LIST, Y_SIZE_DIR_LIST,
                                 overwrite=ARGS.overwrite)

    if ARGS.which == "plot_profile" or ARGS.which == "plot_embolism_counts":
        utilities.update_plot_format()

        if ARGS.output_path is not None:
            if ARGS.output_path == "same":
                PLOT_OUTPUT_LIST = INPUT_JSON_DICT["plots"]["output_paths"]
            else:
                if ARGS.which == "plot_profile":
                    PLOT_OUTPUT_LIST = str.split(ARGS.output_path, " ")
                else:
                    PLOT_OUTPUT_LIST = ARGS.output_path
        else:
            PLOT_OUTPUT_LIST = None

        if ARGS.leaf_names == "same":
            LEAF_NAMES_LIST = INPUT_JSON_DICT["plots"]["leaf_names"]
        elif ARGS.leaf_names:
            LEAF_NAMES_LIST = str.split(ARGS.leaf_names, " ")
        else:
            LEAF_NAMES_LIST = list(range(1, len(MSEQS) + 1))

        if ARGS.which == "plot_profile":
            plot_mseq_profiles(MSEQS, ARGS.show, PLOT_OUTPUT_LIST,
                               LEAF_NAMES_LIST)
        else:
            plot_mseq_embolism_counts(MSEQS, ARGS.show, PLOT_OUTPUT_LIST,
                                      ARGS.tile, LEAF_NAMES_LIST,
                                      ARGS.leaf_embolism_only, ARGS.percent)

    if ARGS.which == "eda_df":
        if ARGS.csv_output_path == "same":
            CSV_OUTPUT_LIST = INPUT_JSON_DICT["eda_df"]["output_path"]
        else:
            CSV_OUTPUT_LIST = str.split(ARGS.csv_output_path, " ")

        if INPUT_JSON_DICT["eda_df"]["options"]["linked_filename"]:
            load_image_objects(LSEQS)

        if ARGS.tiles:
            load_image_objects(MSEQS)

            extract_tiles_eda_df(MSEQS, INPUT_JSON_DICT["eda_df"]["options"],
                                 CSV_OUTPUT_LIST, LSEQS)
        else:
            extract_full_eda_df(MSEQS, INPUT_JSON_DICT["eda_df"]["options"],
                                CSV_OUTPUT_LIST, LSEQS)

    if ARGS.which == "databunch_df":
        load_image_objects(MSEQS)
        load_image_objects(LSEQS)

        if ARGS.csv_output_path == "same":
            CSV_OUTPUT_LIST = INPUT_JSON_DICT["databunch_df"]["output_path"]
        else:
            CSV_OUTPUT_LIST = str.split(ARGS.csv_output_path, " ")

        if ARGS.tiles:
            extract_tiles_databunch_df(
                LSEQS, MSEQS, CSV_OUTPUT_LIST,
                tile_embolism_only=ARGS.tile_embolism_only,
                leaf_embolism_only=ARGS.leaf_embolism_only)
        else:
            extract_full_databunch_df(
                LSEQS, MSEQS, CSV_OUTPUT_LIST, ARGS.leaf_embolism_only)
