import argparse
import json
import logging.config
from os import path
from fastai.vision import *

from src.data.data_model import *
from src.model.fastai_models import FastaiUnetLearner
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


def extract_leaf_images(lseq_list, output_path_list):
    LOGGER.info("Extracting differenced leaf images")
    for lseq, output_path in zip(lseq_list, output_path_list):
        LOGGER.info(f"Differencing images in {lseq.folder_path} and saving "
                    f"to {output_path}")
        lseq.extract_changed_leaves(output_path)


def extract_multipage_mask_images(mseq_list, output_path_list):
    LOGGER.info("Extracting mask images from multipage file")
    for mseq, output_path in zip(mseq_list, output_path_list):
        LOGGER.info(f"Extracting images from: {mseq.mpf_path} and saving "
                    f"to {output_path}")
        mseq.extract_mask_from_multipage(output_path)


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


def plot_mseq_profiles(mseqs,  show, output_path_list):
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
            mseq.plot_profile(show, output_path_list[i])
        else:
            mseq.plot_profile(show)

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
                               embolism_only=False):
    for csv_output_path, lseq, mseq in zip(output_path_list, lseqs, mseqs):

        lseq.link_sequences(mseq)

        LOGGER.info(f"Creating Tile DataBunch DataFrame using "
                    f"{lseq.__class__.__name__} located at {lseq.folder_path} "
                    f"and {mseq.__class__.__name__} located at"
                    f" {mseq.folder_path}")

        _ = lseq.get_tile_databunch_df(mseq, embolism_only,
                                       csv_output_path)


def train_fastai_unet(train_df_paths, val_df_paths, save_path, bs, epochs,
                      lr):

    training_dfs = [pd.read_csv(path, index_col=0) for path in train_df_paths]
    validation_dfs = [pd.read_csv(path, index_col=0) for path in val_df_paths]

    combined_df = combine_and_add_valid(training_dfs, validation_dfs)
    combined_df, folder_name_path = format_databunch_df(
        combined_df, "folder_path", "leaf_names", create_copy=True)

    fai = FastaiUnetLearner()
    fai.prep_fastai_data(combined_df, folder_name_path, bs, plot=False,
                         mask_col_name="masks")
    fai.create_learner()

    fai.train(epochs=epochs, save_path=save_path, lr=lr)


def predict_fastai_unet(lseqs, length_x, length_y, model_pkl_path):
    fai_unet_learner = FastaiUnetLearner(model_pkl_path=model_pkl_path)

    for lseq in lseqs:
        lseq.predict_leaf_sequence(fai_unet_learner, length_x, length_y)


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
        "--leaf_output_path", "-lo",  metavar="\b",
        help="output paths, if the paths are in the input json enter "
             "\"same\"")

    parser_extract_images.add_argument(
        "--mask_output_path", "-mo", metavar="\b",
        help="output paths, if the paths are in the input json enter "
             "\"same\"")

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
        "--leaf_output_path", "-lo",  metavar="\b",
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
    parser_databunch_df.add_argument("--embolism_only", "-eo",
                                     action="store_true")

    parser_train_fastai = subparsers.add_parser(
        "train_fastai", help="train a fastai unet model")
    parser_train_fastai.set_defaults(which="train_fastai")
    parser_train_fastai.add_argument("--batch_size", "-bs", type=int,
                                     metavar="\b", help="batch size")
    parser_train_fastai.add_argument("--save_path", "-sp", type=str,
                                     metavar="\b",
                                     help="path to save model weights and "
                                     "model pickle")
    parser_train_fastai.add_argument("--epochs", "-e", type=int,
                                     metavar="\b", help="number of epochs")
    parser_train_fastai.add_argument("--learning_rate", "-lr", type=float,
                                     metavar="\b", help="learning rate")

    parser_predict_fastai = subparsers.add_parser(
        "predict_fastai", help="train a fastai unet model")
    parser_predict_fastai.set_defaults(which="predict_fastai")
    parser_predict_fastai.add_argument("length_x", type=int,
                                       help="tile x length")
    parser_predict_fastai.add_argument("length_y", type=int,
                                       help="tile y length")
    parser_predict_fastai.add_argument(
        "model_path", help="path to fastai pkl, if the paths are in "
                           "the input json enter \"same\"")

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
                LEAF_OUTPUT_LIST =\
                    INPUT_JSON_DICT["leaves"]["output"]["output_path"]
            else:
                LEAF_OUTPUT_LIST = [ARGS.leaf_output_path]

            extract_leaf_images(LSEQS, LEAF_OUTPUT_LIST)

        if ARGS.mask_output_path is not None:
            if ARGS.mask_output_path == "same":
                MASK_OUTPUT_LIST = \
                    INPUT_JSON_DICT["masks"]["output"]["output_path"]
            else:
                MASK_OUTPUT_LIST = [ARGS.mask_output_path]

            extract_multipage_mask_images(MSEQS, MASK_OUTPUT_LIST)

    if ARGS.which == "extract_tiles":
        if ARGS.leaf_output_path is not None:
            load_image_objects(LSEQS)

            if ARGS.leaf_output_path == "default":
                LEAF_OUTPUT_LIST = None
            elif ARGS.leaf_output_path == "same":
                LEAF_OUTPUT_LIST =\
                    INPUT_JSON_DICT["leaves"]["output"]["output_path"]
            else:
                LEAF_OUTPUT_LIST = str.split(ARGS.leaf_output_path, " ")

            extract_tiles(LSEQS, ARGS.length_x, ARGS.stride_x,
                          ARGS.length_y, ARGS.stride_y, LEAF_OUTPUT_LIST)

        if ARGS.mask_output_path is not None:
            load_image_objects(MSEQS)

            if ARGS.mask_output_path == "default":
                MASK_OUTPUT_LIST = None
            elif ARGS.leaf_output_path == "same":
                MASK_OUTPUT_LIST = \
                    INPUT_JSON_DICT["masks"]["output"]["output_path"]
            else:
                MASK_OUTPUT_LIST = str.split(ARGS.mask_output_path, " ")

            extract_tiles(MSEQS, ARGS.length_x, ARGS.stride_x,
                          ARGS.length_y, ARGS.stride_y, MASK_OUTPUT_LIST)

    if ARGS.which == "plot_profile":
        if ARGS.output_path is not None:
            if ARGS.output_path == "same":
                PLOT_OUTPUT_LIST = INPUT_JSON_DICT["plots"]["output_paths"]
            else:
                PLOT_OUTPUT_LIST = str.split(ARGS.output_path, " ")
        else:
            PLOT_OUTPUT_LIST = None

        plot_mseq_profiles(MSEQS, ARGS.show, PLOT_OUTPUT_LIST)

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
                LSEQS, MSEQS, CSV_OUTPUT_LIST, ARGS.embolism_only)
        else:
            extract_full_databunch_df(
                LSEQS, MSEQS, CSV_OUTPUT_LIST, ARGS.embolism_only)

    if ARGS.which == "train_fastai":
        TRAIN_CSV_INPUT_LIST = INPUT_JSON_DICT["fastai"]["train"]
        VAL_CSV_INPUT_LIST = INPUT_JSON_DICT["fastai"]["validation"]

        train_fastai_unet(TRAIN_CSV_INPUT_LIST, VAL_CSV_INPUT_LIST,
                          save_path=ARGS.save_path, bs=ARGS.batch_size,
                          epochs=ARGS.epochs, lr=ARGS.learning_rate)

    if ARGS.which == "predict_fastai":
        if ARGS.model_path == "same":
            MODEL_PATH = INPUT_JSON_DICT["fastai"]["model"]
        else:
            MODEL_PATH = ARGS.model_path

        load_image_objects(LSEQS)

        predict_fastai_unet(LSEQS, ARGS.length_x, ARGS.length_y,
                            MODEL_PATH)
