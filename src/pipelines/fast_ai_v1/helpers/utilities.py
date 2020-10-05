import argparse

from fastai.vision import *

from src.data.data_model import *
from src.pipelines.fast_ai_1.model.fastai_models import FastaiUnetLearner

LOGGER = logging.getLogger(__name__)


# *============================ DataBunch helpers ============================*
def combine_and_add_valid(train_dfs, valid_dfs):
    if not isinstance(train_dfs, list):
        raise Exception("Please provide the training DataFrame(s) as a list")
    elif not isinstance(valid_dfs, list):
        raise Exception("Please provide the validation DataFrame(s) as a list")

    train_df = pd.concat(train_dfs)
    train_df["is_valid"] = False

    valid_df = pd.concat(valid_dfs)
    valid_df["is_valid"] = True

    return pd.concat([train_df, valid_df])


def format_databunch_df(df_to_update, folder_col_name,
                        leaf_col_name, create_copy=False):
    split_list = df_to_update[folder_col_name].str.split("/").to_list()
    counts_df = pd.DataFrame(split_list).apply(
        lambda x: len(x.unique()), axis=0)

    for i, count in enumerate(counts_df):
        if count > 1:
            break

    if create_copy:
        updated_df = deepcopy(df_to_update)
    else:
        updated_df = df_to_update

    updated_df[leaf_col_name] = updated_df.apply(
        lambda x: os.path.join(*str.split(x[folder_col_name], "/")[i:],
                               x[leaf_col_name]), axis=1)

    common_folder_path = os.path.join("/", *str.split(
        df_to_update[folder_col_name].to_list()[0], "/")[:i], "")

    updated_df.drop(folder_col_name, axis=1, inplace=True)

    return updated_df, common_folder_path


# *================================== main ===================================*
# *---------------------------------- train ----------------------------------*
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


# *--------------------------------- predict ---------------------------------*
def predict_fastai_unet(lseqs, length_x, length_y, model_pkl_path):
    fai_unet_learner = FastaiUnetLearner(model_pkl_path=model_pkl_path)

    for lseq in lseqs:
        lseq.predict_leaf_sequence(fai_unet_learner, length_x, length_y)


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
# *===========================================================================*
