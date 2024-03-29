import logging.config
from ast import literal_eval
from os import path

from src.actions.extraction import *
from src.actions.plots import *
from src.actions.dataset import *
from src.actions.eda import *
from src.actions.prediction import predict_tensorflow
from src.helpers.main_utilities import *

abs_path = path.dirname(path.abspath(__file__))

logging.config.fileConfig(fname=abs_path + "/logging_configuration.ini",
                          defaults={'logfilename': abs_path + "/main.log"},
                          disable_existing_loggers=False)

LOGGER = logging.getLogger(__name__)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

seed = 3141
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)


def main():
    ARGS = parse_arguments()
    if ARGS.interactive:
        ARGS_DICT = interactive_prompt()
    elif ARGS.json_path:
        with open(ARGS.json_path, "r") as JSON_FILE:
            ARGS_DICT = json.load(JSON_FILE)
    else:
        with open(ARGS.filepath_json, "r") as JSON_FILE:
            INPUT_JSON_DICT = json.load(JSON_FILE)

        ARGS_DICT = vars(ARGS)
        ARGS_DICT["leaves"] = {"input": INPUT_JSON_DICT["leaves"]["input"],
                               "format": INPUT_JSON_DICT["leaves"]["format"]}
        ARGS_DICT["masks"] = {"input": INPUT_JSON_DICT["masks"]["input"]}

    LSEQS, MSEQS = create_sequence_objects(ARGS_DICT)

    if ARGS_DICT["which"] == "extract_images":
        if ARGS_DICT["leaf_output_path"] is not None:
            if ARGS_DICT["leaf_output_path"] == "same":
                LEAF_OUTPUT_LIST = \
                    INPUT_JSON_DICT["leaves"]["output"]["output_path"]
            else:
                LEAF_OUTPUT_LIST = str.split(ARGS_DICT["leaf_output_path"],
                                             ";")

            extract_leaf_images(LSEQS, LEAF_OUTPUT_LIST,
                                ARGS_DICT["overwrite"],
                                ARGS_DICT["leaves"]["format"])

        if ARGS_DICT["mask_output_path"] is not None:
            if ARGS_DICT["mask_output_path"] == "same":
                MASK_OUTPUT_LIST = \
                    INPUT_JSON_DICT["masks"]["output"]["output_path"]
            else:
                MASK_OUTPUT_LIST = str.split(ARGS_DICT["mask_output_path"],
                                             ";")

            extract_multipage_mask_images(MSEQS, MASK_OUTPUT_LIST,
                                          ARGS_DICT["overwrite"],
                                          ARGS_DICT["binarise"])

    if ARGS_DICT["which"] == "extract_tiles":
        if ARGS_DICT["leaf_output_path"] is not None:
            load_image_objects(LSEQS)

            if ARGS_DICT["leaf_output_path"] == "default":
                LEAF_OUTPUT_LIST = None
            elif ARGS_DICT["leaf_output_path"] == "same":
                LEAF_OUTPUT_LIST = \
                    INPUT_JSON_DICT["leaves"]["output"]["output_path"]
            else:
                LEAF_OUTPUT_LIST = str.split(ARGS_DICT["leaf_output_path"],
                                             ";")

            extract_tiles(
                LSEQS, ARGS_DICT["length_x"], ARGS_DICT["stride_x"],
                ARGS_DICT["length_y"], ARGS_DICT["stride_y"],
                ARGS_DICT["overlap"], LEAF_OUTPUT_LIST,
                ARGS_DICT["overwrite"], **ARGS_DICT["leaves"]["format"])

        if ARGS_DICT["mask_output_path"] is not None:
            load_image_objects(MSEQS)

            if ARGS_DICT["mask_output_path"] == "default":
                MASK_OUTPUT_LIST = None
            elif ARGS_DICT["mask_output_path"] == "same":
                MASK_OUTPUT_LIST = \
                    INPUT_JSON_DICT["masks"]["output"]["output_path"]
            else:
                MASK_OUTPUT_LIST = str.split(ARGS_DICT["mask_output_path"],
                                             ";")

            extract_tiles(MSEQS, ARGS_DICT["length_x"], ARGS_DICT["stride_x"],
                          ARGS_DICT["length_y"], ARGS_DICT["stride_y"],
                          ARGS_DICT["overlap"], MASK_OUTPUT_LIST,
                          ARGS_DICT["overwrite"])

    if ARGS_DICT["which"] == "trim_sequence":
        if ARGS_DICT["mask"]:
            SEQS = MSEQS
            FORMAT_DICT = {}
        else:
            SEQS = LSEQS
            FORMAT_DICT = ARGS_DICT["leaves"]["format"]

        if ARGS_DICT["x_size_dir"] == "same":
            X_SIZE_DIR_LIST = INPUT_JSON_DICT["trim"]["x_size_dir"]
            X_SIZE_DIR_LIST = [tuple(X_SIZE_DIR) if X_SIZE_DIR else None for
                               X_SIZE_DIR in X_SIZE_DIR_LIST]
        elif not ARGS_DICT["x_size_dir"]:
            X_SIZE_DIR_LIST = None
        else:
            X_SIZE_DIR_LIST = [literal_eval(x_size_dir) for x_size_dir in
                               ARGS_DICT["x_size_dir"].split(";")]

        if ARGS_DICT["y_size_dir"] == "same":
            Y_SIZE_DIR_LIST = INPUT_JSON_DICT["trim"]["y_size_dir"]
            Y_SIZE_DIR_LIST = [tuple(Y_SIZE_DIR) if Y_SIZE_DIR else None for
                               Y_SIZE_DIR in Y_SIZE_DIR_LIST]
        elif not ARGS_DICT["y_size_dir"]:
            Y_SIZE_DIR_LIST = [None for _ in range(len(X_SIZE_DIR_LIST))]
        else:
            Y_SIZE_DIR_LIST = [literal_eval(y_size_dir) for y_size_dir in
                               ARGS_DICT["y_size_dir"].split(";")]

        if not X_SIZE_DIR_LIST:
            X_SIZE_DIR_LIST = [None for _ in range(len(Y_SIZE_DIR_LIST))]

        if ARGS_DICT["y_size_dir"] or ARGS_DICT["x_size_dir"]:
            load_image_objects(SEQS)
            trim_sequence_images(SEQS, X_SIZE_DIR_LIST, Y_SIZE_DIR_LIST,
                                 overwrite=ARGS_DICT["overwrite"],
                                 **FORMAT_DICT)

    if (ARGS_DICT["which"] == "plot_profile" or
            ARGS_DICT["which"] == "plot_embolism_counts"):
        utilities.update_plot_format()

        if ARGS_DICT["output_path"] is not None:
            if ARGS_DICT["output_path"] == "same":
                PLOT_OUTPUT_LIST = INPUT_JSON_DICT["plots"]["output_paths"]
            else:
                if ARGS_DICT["which"] == "plot_profile":
                    PLOT_OUTPUT_LIST = str.split(ARGS_DICT["output_path"], ";")
                else:
                    PLOT_OUTPUT_LIST = ARGS_DICT["output_path"]
        else:
            PLOT_OUTPUT_LIST = None

        if ARGS_DICT["leaf_names"] == "same":
            LEAF_NAMES_LIST = INPUT_JSON_DICT["plots"]["leaf_names"]
        elif ARGS_DICT["leaf_names"]:
            LEAF_NAMES_LIST = str.split(ARGS_DICT["leaf_names"], ";")
        else:
            LEAF_NAMES_LIST = list(range(1, len(MSEQS) + 1))

        if ARGS_DICT["which"] == "plot_profile":
            plot_mseq_profiles(MSEQS, ARGS_DICT["show"], PLOT_OUTPUT_LIST,
                               LEAF_NAMES_LIST)
        else:
            plot_mseq_embolism_counts(MSEQS, ARGS_DICT["show"],
                                      PLOT_OUTPUT_LIST,
                                      ARGS_DICT["tile"], LEAF_NAMES_LIST,
                                      ARGS_DICT["leaf_embolism_only"],
                                      ARGS_DICT["percent"])

    if ARGS_DICT["which"] == "eda_df":
        if not ARGS.interactive and not ARGS.json_path:
            ARGS_DICT["eda_df_options"] = INPUT_JSON_DICT["eda_df"]["options"]
        if ARGS_DICT["csv_output_path"] == "same":
            CSV_OUTPUT_LIST = INPUT_JSON_DICT["eda_df"]["output_path"]
        else:
            CSV_OUTPUT_LIST = str.split(ARGS_DICT["csv_output_path"], ";")

        if ARGS_DICT["eda_df_options"]["linked_filename"]:
            load_image_objects(LSEQS)

        if ARGS_DICT["tiles"]:
            load_image_objects(MSEQS)

            extract_tiles_eda_df(MSEQS, ARGS_DICT["eda_df_options"],
                                 CSV_OUTPUT_LIST, LSEQS)
        else:
            extract_full_eda_df(MSEQS, ARGS_DICT["eda_df_options"],
                                CSV_OUTPUT_LIST, LSEQS)

    if ARGS_DICT["which"] == "databunch_df":
        load_image_objects(MSEQS)
        load_image_objects(LSEQS)

        if ARGS_DICT["csv_output_path"] == "same":
            CSV_OUTPUT_LIST = INPUT_JSON_DICT["databunch_df"]["output_path"]
        else:
            CSV_OUTPUT_LIST = str.split(ARGS_DICT["csv_output_path"], ";")

        if ARGS_DICT["tiles"]:
            extract_tiles_databunch_df(
                LSEQS, MSEQS, CSV_OUTPUT_LIST,
                tile_embolism_only=ARGS_DICT["tile_embolism_only"],
                leaf_embolism_only=ARGS_DICT["leaf_embolism_only"])
        else:
            extract_full_databunch_df(
                LSEQS, MSEQS, CSV_OUTPUT_LIST, ARGS_DICT["leaf_embolism_only"])

    if ARGS_DICT["which"] == "predict":
        load_image_objects(LSEQS)

        if ARGS_DICT["csv_path"]:
            load_image_objects(MSEQS)

        leaf_shape = tuple([int(dim) for dim in
                            ARGS_DICT["leaf_shape"].split(";")])
        predict_tensorflow(LSEQS, model_weight_path=ARGS_DICT["model_path"],
                           leaf_shape=leaf_shape,
                           cr_csv_list=ARGS_DICT["csv_path"],
                           mseqs=MSEQS,
                           threshold=ARGS_DICT["threshold"],
                           format_dict=ARGS_DICT["leaves"]["format"])

    if ARGS_DICT["which"] == "create_dataset":
        load_image_objects(LSEQS)
        load_image_objects(MSEQS)

        if ARGS_DICT["lolo"]:
            if len(LSEQS) == 1:
                raise ValueError("You need to provide at least two leaf "
                                 "sequences to create a leave one leaf out "
                                 "test set.")

            ARGS_DICT["lolo"] = int(ARGS_DICT["lolo"]) - 1

            if ARGS_DICT["test_split"] > 0:
                LOGGER.warning("You provided both a leaf to leave out and a "
                               "test split for a test set. The test set split "
                               "will now be set to 0.")

                ARGS_DICT["test_split"] = 0
        else:
            ARGS_DICT["lolo"] = None

        extract_dataset(LSEQS, MSEQS, ARGS_DICT["dataset_path"],
                        ARGS_DICT["downsample_split"], ARGS_DICT["test_split"],
                        ARGS_DICT["val_split"], ARGS_DICT["lolo"])

    if ARGS_DICT["which"] == "augment_dataset":
        load_image_objects(LSEQS)
        load_image_objects(MSEQS)

        augment_dataset(LSEQS[0], MSEQS[0], **ARGS_DICT["leaves"]["format"])


if __name__ == "__main__":
    main()
