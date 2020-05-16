import argparse
import json
import logging.config
from os import path

from src.data.data_model import *

abs_path = path.dirname(path.abspath(__file__))

logging.config.fileConfig(fname=abs_path + "/logging_configuration.ini",
                          defaults={'logfilename': abs_path + "/main.log"},
                          disable_existing_loggers=False)
LOGGER = logging.getLogger(__name__)


def create_sequence_objects(json_path: str):
    lseqs = []
    mseqs = []

    with open(json_path, "r") as json_file:
        sequence_input = json.load(json_file)

    leaf_keys = list(sequence_input["leaves"].keys())
    if sequence_input["leaves"] is not None:
        for vals in zip(*list(sequence_input["leaves"].values())):
            kwargs = {key: val for key, val in zip(leaf_keys, vals)}
            lseqs.append(LeafSequence(**kwargs))

    mask_keys = list(sequence_input["masks"].keys())
    if sequence_input["masks"] is not None:
        for vals in zip(*list(sequence_input["masks"].values())):
            kwargs = {key: val for key, val in zip(mask_keys, vals)}
            mseqs.append(MaskSequence(**kwargs))

    return lseqs, mseqs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Perform operations using the "
                                     "plant-image-segmentation code base")
    parser.add_argument("json_path", type=str,
                        help="path to a JSON file with the required "
                             "parameters to create LeafSequence and "
                             "MaskSequence objects")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_arguments()
    LSEQS, MSEQS = create_sequence_objects(ARGS.json_path)
