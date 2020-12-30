import shutil

from sklearn.model_selection import train_test_split

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

    train_dir = base_dir.joinpath("train")
    val_dir = base_dir.joinpath("val")
    test_dir = base_dir.joinpath("test")

    path_list = [train_dir, val_dir, test_dir]

    for path in path_list:
        for folder in ["embolism", "no-embolism"]:
            path.joinpath(folder, "leaves").mkdir(parents=True,
                                                  exist_ok=True)
            path.joinpath(folder, "masks").mkdir(parents=True,
                                                 exist_ok=True)


    base_dir.joinpath("not_used", "masks").mkdir(parents=True,
                                                 exist_ok=True)
    base_dir.joinpath("not_used", "leaves").mkdir(parents=True,
                                                  exist_ok=True)

    return base_dir


def create_train_dataset(lseqs, mseqs, dest_root_path) -> [str, str]:
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

        mask_chip_path = Path(mseq.image_objects[0].file_list[0])
        mask_chip_folder = Path(*mask_chip_path.parts[:-1])

        leaf_chip_path = Path(lseq.image_objects[0].file_list[0])
        leaf_chip_folder = Path(*leaf_chip_path.parts[:-1])

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


    #Note: All leaf and mask tiles must have the same file extension
    # Get the extension using the filenames of the chips of the last chip
    # paths from the above loop
    mask_file_ext = "*." + str(mask_chip_path.parts[-1]).rsplit(".")[1]
    leaf_file_ext = "*." + str(leaf_chip_path.parts[-1]).rsplit(".")[1]

    return [leaf_file_ext, mask_file_ext]


def downsample_dataset(dataset_root_path, filename_patterns,
                       non_embolism_size=0.5):
    if not isinstance(dataset_root_path, Path):
        dataset_root_path = Path(dataset_root_path)

    train_emb_path = dataset_root_path.joinpath("train", "embolism")
    train_no_emb_path = dataset_root_path.joinpath("train", "no-embolism")

    # Getting all the embolism and non-embolism images in the dataset
    ne_leaves = sorted([f for f in glob(
        str(train_no_emb_path.joinpath("leaves", filename_patterns[0])),
        recursive=True)])
    ne_masks = sorted([f for f in glob(
        str(train_no_emb_path.joinpath("masks", filename_patterns[1])),
        recursive=True)])

    e_leaves = sorted([f for f in glob(
        str(train_emb_path.joinpath("leaves", filename_patterns[0])),
        recursive=True)])
    e_masks = sorted([f for f in glob(
        str(train_emb_path.joinpath("masks", filename_patterns[1])),
        recursive=True)])

    # randomly selected non embolism samples to ignore
    # if odd, then chosen items get the extra sample
    ignored_masks, chosen_masks, ignored_leaves, chosen_leaves = \
        train_test_split(ne_masks, ne_leaves, test_size=non_embolism_size,
                         random_state=3141)

    # down sample by moving the non-embolism samples
    not_used_path = dataset_root_path.joinpath("not_used")

    # add the chip type (-2) and name (-1) to the not_used_path to create new
    # location | requires default folder structure
    _ = list(map(lambda x:
                 shutil.move(x, not_used_path.joinpath(*Path(x).parts[-2:])),
                 ignored_masks + ignored_leaves))

    LOGGER.info(f"Downsampled {len(ignored_leaves)} ("
                f"{len(ignored_leaves)/len(ignored_masks+chosen_masks)})% "
                f"non-embolism images")

    return [e_leaves, e_masks], [ne_leaves, ne_masks]
