import shutil

from sklearn.model_selection import train_test_split

from src.data.data_model import *
from typing import Union

from pathlib import Path


# *============================= create dataset ==============================*
def create_dataset_structure(base_dir: Union[Path, str]) -> None:
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


def move_data(lseqs, mseqs, dest_root_path,
                         dest_folder="train") -> [str, str]:
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
                dest_root_path.joinpath(dest_folder, "embolism", "masks", x)))

        LOGGER.info("Moving non-embolism masks")
        embolism_df[~embolism_df.has_embolism].names.map(
            lambda x: shutil.copyfile(
                mask_chip_folder.joinpath(x),
                dest_root_path.joinpath(dest_folder, "no-embolism", "masks", x)))

        # Leaves
        LOGGER.info("Moving embolism leaves")
        embolism_df[embolism_df.has_embolism].links.map(
            lambda x: shutil.copyfile(
                leaf_chip_folder.joinpath(x),
                dest_root_path.joinpath(dest_folder, "embolism", "leaves", x)))

        LOGGER.info("Moving non-embolism leaves")
        embolism_df[~embolism_df.has_embolism].links.map(
            lambda x: shutil.copyfile(
                leaf_chip_folder.joinpath(x),
                dest_root_path.joinpath(dest_folder, "no-embolism", "leaves", x)))


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

    percent_moved = len(ignored_leaves) / len(ignored_masks + chosen_masks)
    LOGGER.info(f"Downsampled by {len(ignored_leaves)} "
                f"({round(percent_moved * 100)})% non-embolism images")

    return [e_leaves, e_masks], [chosen_leaves, chosen_masks]


def split_dataset(dataset_root_path, embolism_objects, non_embolism_objects,
                  test_split=0.2, val_split=0.2):
    """

    :param dataset_root_path:
    :param embolism_objects: a list containing paths to embolism masks and
    leaves; leaves at item 0 and masks at item 1
    :param non_embolism_objects:  list containing paths to non-embolism masks
    and leaves; leaves at item 0 and masks at item 1
    :param test_size:
    :param val_size:
    :return:
    """
    e_leaves = embolism_objects[0]
    e_masks = embolism_objects[1]

    ne_leaves = non_embolism_objects[0]
    ne_masks = non_embolism_objects[1]

    total_size = len(e_leaves + ne_leaves)
    val_size = 0
    test_size = 0

    if not isinstance(dataset_root_path, Path):
        dataset_root_path = Path(dataset_root_path)

    # Splitting test set and (train + val) set
    if test_split > 0:
        test_path = dataset_root_path.joinpath("test")

        # Embolism
        # split testset and keep the remaining files together to be split again
        e_train_val_masks, e_test_masks, e_train_val_leaves, e_test_leaves = \
            train_test_split(e_masks, e_leaves, test_size=test_split,
                             random_state=3141)

        # Non-embolism
        ne_train_val_masks, ne_test_masks, ne_train_val_leaves, \
        ne_test_leaves = train_test_split(ne_masks, ne_leaves,
                                          test_size=test_split,
                                          random_state=3141)

        # Move files
        # Requires default folder structure
        _ = list(map(lambda x: shutil.move(
            x, test_path.joinpath(*Path(x).parts[-3:])),
                     e_test_masks + e_test_leaves + ne_test_masks +
                     ne_test_leaves))

        test_size = len(e_test_leaves + ne_test_leaves)
        percent_moved = (test_size / total_size) * 100
        LOGGER.info(f"Moved {test_size} "
                    f"({round(percent_moved)} %) samples to the test folder")

    else:
        # If no test set, then split all images between train and val
        e_train_val_masks = e_masks
        e_train_val_leaves = e_leaves

        ne_train_val_masks = ne_masks
        ne_train_val_leaves = ne_leaves

    # split train_val set into train and val set
    if val_split > 0:
        val_path = dataset_root_path.joinpath("val")
        # Getting val set, % of train set after test set has been removed
        # Embolism
        _, e_val_masks, _, e_val_leaves = \
            train_test_split(e_train_val_masks, e_train_val_leaves,
                             test_size=val_split, random_state=3141)

        # Non-embolism
        ne_train_masks, ne_val_masks, ne_train_leaves, ne_val_leaves = \
            train_test_split(ne_train_val_masks, ne_train_val_leaves,
                             test_size=val_split, random_state=3141)

        val_size = len(e_val_leaves + ne_val_leaves)
        percent_moved = (val_size /
                         len(e_train_val_leaves + ne_train_val_leaves)) * 100
        LOGGER.info(
            f"Moved {val_size} ("
            f"{round(percent_moved)} %) of the remaining train samples to "
            f"the val folder")

        # Move files
        _ = list(map(lambda x: shutil.move(
            x, val_path.joinpath(*Path(x).parts[-3:])),
                     e_val_masks + e_val_leaves + ne_val_masks + ne_val_leaves))

    train_size = total_size - val_size - test_size
    LOGGER.info(
        f"Summary: (% of total number of images) "
        f"\nTraining set size   :  {train_size} "
        f"({round((train_size/total_size) * 100)  }%)"
        f"\nValidation set size :  {val_size} "
        f"({round((val_size/total_size)* 100)}%) "
        f"\nTest set size       :  {test_size} "
        f"({round((test_size/total_size) * 100)}%)")


# *============================ package __main__ =============================*
def extract_dataset(lseqs: [LeafSequence], mseqs: [MaskSequence],
                    dataset_path: Union[Path, str],
                    downsample_split: float, test_split: float,
                    val_split: float) -> None:
    """

    :param lseqs:
    :param mseqs:
    :param dataset_path:
    :param downsample_split:
    :param test_split:
    :param val_split:
    :return:
    """
    # will create a structure iff one does not exist in the correct
    # format at the specified path
    create_dataset_structure(dataset_path)
    filename_patterns = move_data(lseqs, mseqs, dataset_path)

    # non_emb_list will contain the filenames of chosen non-embolism images
    emb_list, non_emb_list = downsample_dataset(dataset_path,
                                                filename_patterns,
                                                downsample_split)

    split_dataset(dataset_path, emb_list, non_emb_list, test_split, val_split)
# *===========================================================================*
