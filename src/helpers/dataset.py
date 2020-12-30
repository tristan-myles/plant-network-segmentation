import shutil

from src.data.data_model import *
from typing import Union

from pathlib import Path


# *============================= create dataset ==============================*
def create_dataset_structure(base_dir: Union[Path, str]) -> Path:
    """
    Creates a skeleton dataset structure
    :param base_dir: the directory where the dataset should be created,
    in either a pathlib Path or srt format
    :return: None
    """
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    base_dir = base_dir.joinpath("dataset")

    train_dir = base_dir.joinpath("train")
    val_dir = base_dir.joinpath("val")
    test_dir = base_dir.joinpath("test")

    path_list = [train_dir, val_dir, test_dir]

    for path in path_list:
        path.joinpath("embolism", "leaves").mkdir(parents=True, exist_ok=True)
        path.joinpath("embolism", "masks").mkdir(parents=True, exist_ok=True)
        path.joinpath("no-embolism", "leaves").mkdir(parents=True,
                                                     exist_ok=True)
        path.joinpath("no-embolism", "masks").mkdir(parents=True,
                                                    exist_ok=True)

    base_dir.joinpath("not_used").mkdir(parents=True, exist_ok=True)

    return base_dir


def create_train_dataset(lseqs, mseqs, dest_root_path) -> None:
    """
    Populates the train folder in the dataset folder, where the dataset 
    folder and its constituents were created using the create_dataset_structure
    function of this module.


    :param lseqs: list of leaf sequence objects
    :param mseqs: list of mask sequence objects
    :param dest_root_path:
    :return: None

    .. note:: This function requires both leaves and masks to be in the same
              root directory
    """
    if not isinstance(dest_root_path, Path):
        dest_root_path = Path(dest_root_path)

    for lseq, mseq in zip(lseqs, mseqs):
        lseq.load_extracted_images()
        mseq.load_extracted_images()

        lseq.link_sequences(mseq)
        mseq.link_sequences(lseq)

        embolism_df = mseq.get_tile_eda_df({
            "linked_filename": True,
            "unique_range": False,
            "embolism_percent": True,
            "intersection": False,
            "has_embolism": True})

        mask_chip_folder, _ = mseq.image_objects[0].file_list[0].rsplit("/", 1)
        mask_chip_folder = Path(mask_chip_folder)

        leaf_chip_folder, _ = lseq.image_objects[0].file_list[0].rsplit("/", 1)
        leaf_chip_folder = Path(leaf_chip_folder)

        # Masks
        LOGGER.info("Moving embolism masks")
        embolism_df[embolism_df.has_embolism].names.map(
            lambda x: shutil.copyfile(
                mask_chip_folder.joinpath(x),
                dest_root_path.joinpath("train", "embolism", "masks", x)))

        LOGGER.info("Moving non-embolism masks")
        embolism_df[~embolism_df.has_embolism].names.map(
            lambda x: shutil.copyfile(
                mask_chip_folder.joinpath(x),
                dest_root_path.joinpath("train", "no-embolism", "masks", x)))

        # Leaves
        LOGGER.info("Moving embolism leaves")
        embolism_df[embolism_df.has_embolism].links.map(
            lambda x: shutil.copyfile(
                leaf_chip_folder.joinpath(x),
                dest_root_path.joinpath("train", "embolism", "leaves", x)))

        LOGGER.info("Moving non-embolism leaves")
        embolism_df[~embolism_df.has_embolism].links.map(
            lambda x: shutil.copyfile(
                leaf_chip_folder.joinpath(x),
                dest_root_path.joinpath("train", "no-embolism", "leaves", x)))
