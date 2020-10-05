import json
import logging.config
from ast import literal_eval
from os import path

from src.helpers.extraction import *
from src.helpers.plots import *
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


def main():
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


if __name__ == "__main__":
    main()
