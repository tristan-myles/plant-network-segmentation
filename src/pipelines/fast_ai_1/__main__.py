import argparse
import json
import logging.config
from os import path

from src.data.data_model import *
from src.model.fastai_models import FastaiUnetLearner
from src.pipelines.fast_ai_1.helpers.utilities import (combine_and_add_valid,
                                                       format_databunch_df)

abs_path = path.dirname(path.abspath(__file__))

logging.config.fileConfig(fname=abs_path + "/../../logging_configuration.ini",
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


def train_fastai_unet(train_df_paths, save_path, bs, epochs, lr,
                      val_df_paths=None, unfreeze_type=None,
                      model_weights_path=None, **kwargs):
    training_dfs = [pd.read_csv(path, index_col=0) for path in train_df_paths]

    if val_df_paths:
        validation_dfs = [pd.read_csv(path, index_col=0) for path in
                          val_df_paths]
        combined_df = combine_and_add_valid(training_dfs, validation_dfs)
    else:
        combined_df = pd.concat(training_dfs)

    combined_df, folder_name_path = format_databunch_df(
        combined_df, "folder_path", "leaf_name", create_copy=True)

    fai = FastaiUnetLearner()

    if val_df_paths:
        fai.prep_fastai_data(combined_df, folder_name_path, bs, plot=False,
                             mask_col_name="mask_path",
                             **kwargs)
    else:
        fai.prep_fastai_data(combined_df, folder_name_path, bs, plot=False,
                             mask_col_name="mask_path",
                             split_func=ItemList.split_by_rand_pct,
                             **kwargs)

    fai.create_learner()

    if model_weights_path:
        fai.load_weights(model_weights_path)

    fai.train(epochs=epochs, save_path=save_path, lr=lr,
              unfreeze_type=unfreeze_type)


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
    parser_train_fastai.add_argument(
        "--unfreeze_type", "-ut", metavar="\b",
        help="whether the model needs to be unfrozen before training, "
             "the options that follow refer to the layer you want to "
             "unfreeze: 'all', 'last', or the number you want to freeze up "
             "to")
    parser_train_fastai.add_argument(
        "--model_weights_path", "-mwp",
        help="path to fastai model weights (complete path excluding .pth) to "
             "retrain, if the paths are in the input json enter \"same\"")

    parser_predict_fastai = subparsers.add_parser(
        "predict_fastai", help="train a fastai unet model")
    parser_predict_fastai.set_defaults(which="predict_fastai")
    parser_predict_fastai.add_argument("length_x", type=int,
                                       help="tile x length")
    parser_predict_fastai.add_argument("length_y", type=int,
                                       help="tile y length")
    parser_predict_fastai.add_argument(
        "model_path", help="path to fastai pkl (complete path), if the paths "
                           "are in the input json enter \"same\"")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_arguments()

    LOGGER.debug(ARGS.which)
    with open(ARGS.json_path, "r") as JSON_FILE:
        INPUT_JSON_DICT = json.load(JSON_FILE)

    if ARGS.which != "train_fastai":
        LSEQS, MSEQS = create_sequence_objects(INPUT_JSON_DICT)

    if ARGS.which == "train_fastai":
        TRAIN_CSV_INPUT_LIST = INPUT_JSON_DICT["fastai"]["train"]
        VAL_CSV_INPUT_LIST = INPUT_JSON_DICT["fastai"]["validation"]

        if ARGS.unfreeze_type:
            if ARGS.unfreeze_type != "all" and ARGS.unfreeze_type != "last":
                UNFREEZE_TYPE = int(ARGS.unfreeze_type)
            else:
                UNFREEZE_TYPE = ARGS.unfreeze_type
        else:
            UNFREEZE_TYPE = None

        if VAL_CSV_INPUT_LIST:
            train_fastai_unet(TRAIN_CSV_INPUT_LIST, save_path=ARGS.save_path,
                              bs=ARGS.batch_size, epochs=ARGS.epochs,
                              lr=ARGS.learning_rate,
                              val_df_paths=VAL_CSV_INPUT_LIST,
                              unfreeze_type=UNFREEZE_TYPE,
                              model_weights_path=ARGS.model_weights_path)
        else:
            train_fastai_unet(TRAIN_CSV_INPUT_LIST,
                              save_path=ARGS.save_path, bs=ARGS.batch_size,
                              epochs=ARGS.epochs, lr=ARGS.learning_rate,
                              unfreeze_type=UNFREEZE_TYPE,
                              model_weights_path=ARGS.model_weights_path,
                              seed=314)

    if ARGS.which == "predict_fastai":
        if ARGS.model_path == "same":
            MODEL_PATH = INPUT_JSON_DICT["fastai"]["model"]
        else:
            MODEL_PATH = ARGS.model_path

        load_image_objects(LSEQS)

        predict_fastai_unet(LSEQS, ARGS.length_x, ARGS.length_y,
                            MODEL_PATH)
