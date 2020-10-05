import logging.config
from os import path

from src.helpers.utilities import create_sequence_objects, load_image_objects
from src.pipelines.fast_ai_v1.helpers.utilities import *

abs_path = path.dirname(path.abspath(__file__))

logging.config.fileConfig(fname=abs_path + "/../../logging_configuration.ini",
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


if __name__ == "__main__":
    main()
