import shutil

from sklearn.model_selection import train_test_split

from src.data.data_model import *
from src.helpers.utilities import create_subfolders
from typing import Union
import imgaug.augmenters as iaa

from pathlib import Path

import random
random.seed(3141)


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
            create_subfolders(path, folder)

    create_subfolders(base_dir, "not_used")


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
        LOGGER.info("Moving masks")
        embolism_df[embolism_df.has_embolism].names.map(
            lambda x: shutil.copyfile(
                mask_chip_folder.joinpath(x),
                dest_root_path.joinpath(dest_folder, "embolism", "masks", x)))

        embolism_df[~embolism_df.has_embolism].names.map(
            lambda x: shutil.copyfile(
                mask_chip_folder.joinpath(x),
                dest_root_path.joinpath(dest_folder, "no-embolism", "masks", x)))

        # Leaves
        LOGGER.info("Moving leaves")
        embolism_df[embolism_df.has_embolism].links.map(
            lambda x: shutil.copyfile(
                leaf_chip_folder.joinpath(x),
                dest_root_path.joinpath(dest_folder, "embolism", "leaves", x)))

        embolism_df[~embolism_df.has_embolism].links.map(
            lambda x: shutil.copyfile(
                leaf_chip_folder.joinpath(x),
                dest_root_path.joinpath(dest_folder, "no-embolism",
                                        "leaves", x)))

        LOGGER.info(f"Moved {len(embolism_df)} images to "
                    f"{dest_root_path.joinpath(dest_folder, '*')}")

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
        f"Summary: (% of total number of images used in this split) "
        f"\nTraining set size   :  {train_size} "
        f"({round((train_size/total_size) * 100)  }%)"
        f"\nValidation set size :  {val_size} "
        f"({round((val_size/total_size)* 100)}%) "
        f"\nTest set size       :  {test_size} "
        f"({round((test_size/total_size) * 100)}%)")


# *---------------------------- package __main__ -----------------------------*
def extract_dataset(lseqs: [LeafSequence], mseqs: [MaskSequence],
                    dataset_path: Union[Path, str],
                    downsample_split: float, test_split: float,
                    val_split: float, lolo: int=None) -> None:
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

    if isinstance(lolo, int):
        # isolate the leaf to leave out
        lseq_lolo = [lseqs[-lolo]]
        mseq_lolo = [mseqs[-lolo]]

        _ = move_data(lseq_lolo, mseq_lolo, dataset_path, "test")

        # remove the leaf to leave out from seqs
        del lseqs[lolo]
        del mseqs[lolo]

    filename_patterns = move_data(lseqs, mseqs, dataset_path)

    # non_emb_list will contain the filenames of chosen non-embolism images
    emb_list, non_emb_list = downsample_dataset(dataset_path,
                                                filename_patterns,
                                                downsample_split)

    split_dataset(dataset_path, emb_list, non_emb_list, test_split, val_split)


# *============================= augment dataset =============================*
# *----------------------------- transformations -----------------------------*
def flip_flop(dual_channel: np.array, orientation: str,
              seed: int = 3141) -> np.array:
    """

    :param dual_channel:
    :param orientation:
    :param seed:
    :return:
    """
    if orientation == "horizontal":
        flip_hr = iaa.Fliplr(seed=seed)
        flipped_images = flip_hr.augment_image(dual_channel)
    elif orientation == "vertical":
        flip_vr = iaa.Flipud(seed=seed)
        flipped_images = flip_vr.augment_image(dual_channel)
    else:
        raise ValueError("please provide either 'horizontal' or 'vertical as "
                         "the orientation'")

    return flipped_images


def translate_img(dual_channel: np.array, x: float, y: float,
                  seed:int = 3141) -> np.array:
    """

    :param dual_channel:
    :param x:
    :param y:
    :param seed:
    :return:
    """
    rotate = iaa.Affine(translate_percent=(x, y), seed=seed)
    return rotate.augment_image(dual_channel)


def rotate_img(dual_channel: np.array, l: float, r: float,
               seed: int = 3141) -> np.array:
    """

    :param dual_channel:
    :param l:
    :param r:
    :param seed:
    :return:
    """
    rotate = iaa.Affine(rotate=(l, r), seed=seed)
    return rotate.augment_image(dual_channel)


def shear_img(dual_channel: np.array, l: float, r: float,
              seed: int = 3141) -> np.array:
    """

    :param dual_channel:
    :param l:
    :param r:
    :param seed:
    :return:
    """
    # Shear in degrees
    shear = iaa.Affine(shear=(l, r), seed=seed)
    return shear.augment_image(dual_channel)


def crop_img(dual_channel: np.array, v: float, h: float,
             seed: int = 3141) -> np.array:
    """

    :param dual_channel:
    :param v:
    :param h:
    :param seed:
    :return:
    """
    crop = iaa.Crop(percent=(v, h), seed=seed)
    return crop.augment_image(dual_channel)


def zoom_in_out(dual_channel: np.array, x: float, y: float,
                seed: int = 3141) -> np.array:
    """

    :param dual_channel:
    :param x:
    :param y:
    :param seed:
    :return:
    """
    scale_im = iaa.Affine(scale={"x": x, "y": y}, seed=seed)
    return scale_im.augment_image(dual_channel)


# *--------------------------------- helpers ---------------------------------*
def stack_images(leaf_image: np.array, mask_image: np.array) -> np.array:
    """

    :param leaf_image:
    :param mask_image:
    :return:
    """
    # Change back to 0 if not float
    # Stack leaf and mask together so that they can be augmented uniformly
    leaf_image = leaf_image[:, :, np.newaxis]
    mask_image = mask_image[:, :, np.newaxis]

    # leaf first
    dual_channel = np.concatenate((leaf_image, mask_image), axis=2)

    return dual_channel


def save_image(image: np.array, leaf_path: str, mask_path: str,
               aug_type: str) -> None:
    """

    :param image:
    :param leaf_path:
    :param mask_path:
    :param aug_type:
    :return:
    """
    old_paths = [leaf_path, mask_path]
    # ["leaf", "mask"]
    new_paths = ["", ""]

    for i, path in enumerate(old_paths):
        # requires default dataset folder structure
        path_list = list(Path(path).parts)
        path_list[-3] = "augmented"

        # requires default naming
        filename, ext = path_list[-1].rsplit(".", 1)

        # add the description to the file name, after the image, and tile
        # number to keep images tiles grouped
        filename = ".".join(["_".join([filename, aug_type]), ext])

        path_list[-1] = filename

        new_paths[i] = Path(*path_list)

    # leaf is first in stacked array
    cv2.imwrite(str(new_paths[0]), image[:, :, 0])
    cv2.imwrite(str(new_paths[1]), image[:, :, 1])


def augment_image(stacked_image: np.array, df: pd.DataFrame, aug_type: str,
                  index: int, counts: [int, int], leaf_path: str,
                  mask_path: str, func, **kwargs) -> [int, int]:
    """

    :param stacked_image:
    :param df:
    :param aug_type:
    :param index:
    :param counts:
    :param leaf_path:
    :param mask_path:
    :param func:
    :param kwargs:
    :return:
    """
    image = func(stacked_image, **kwargs)

    # only save an image if it has an embolism
    # binary segmentation problem so we know that if there are two pixel
    # intensities there are embolisms

    if len(np.unique(image[:, :, 1])) > 1:
        save_image(image, leaf_path, mask_path, aug_type)

        df[aug_type][index] = ', '.join(
            [f'{k}: {v}' for k, v in kwargs.items()])

        counts[0] += 1
    else:
        counts[1] += 1

    return counts


def augmentation_algorithm(dual_channel: np.array, aug_df: pd.DataFrame,
                           i: int, leaf_path: str, mask_path: str,
                           counts: [int, int]) -> (pd.DataFrame, [int, int]):
    """

    :param dual_channel:
    :param aug_df:
    :param i:
    :param leaf_path:
    :param mask_path:
    :param counts:
    :return:
    """
    # P(flip) = 0.35
    if random.random() < 0.35:
        # P(H | flip) = 0.5 | P(V | flip) = 0.5
        if random.random() < 0.5:
            orientation = "horizontal"
        else:
            orientation = "vertical"

        counts = augment_image(dual_channel, aug_df, "flip", i, counts,
                               leaf_path, mask_path, flip_flop,
                               orientation=orientation)

    # P(translate) = 0.35
    if random.random() < 0.35:
        # zoom in and out between -25% and 25%
        x_per = round(random.uniform(-0.25, 0.25), 2)
        y_per = round(random.uniform(-0.25, 0.25), 2)

        counts = augment_image(dual_channel, aug_df, "translate", i, counts,
                               leaf_path, mask_path, translate_img, x=x_per,
                               y=y_per)

    # P(zoom) = 0.35
    if random.random() < 0.35:
        # zoom in and out between 150% and 50%
        x_per = round(random.uniform(1.5, 0.5), 2)
        y_per = round(random.uniform(1.5, 0.5), 2)

        counts = augment_image(dual_channel, aug_df, "zoom", i, counts,
                               leaf_path, mask_path, zoom_in_out,
                               x=x_per, y=y_per)

    # P(crop) = 0.35
    if random.random() < 0.35:
        # crop between 5% and 30% of the image
        v_per = round(random.uniform(0.05, 0.3), 2)
        h_per = round(random.uniform(0.05, 0.3), 2)

        counts = augment_image(dual_channel, aug_df, "crop", i, counts,
                               leaf_path, mask_path, crop_img, v=v_per,
                               h=h_per)

    # P(rotate) = 0.35
    if random.random() < 0.35:
        # l element (-90;0) and r element (0;90) (degrees)
        l_deg = round(random.random() * -90)
        r_deg = round(random.random() * 90)

        counts = augment_image(dual_channel, aug_df, "rotate", i, counts,
                               leaf_path, mask_path, rotate_img, l=l_deg,
                               r=r_deg)

    # P(sheer) = 0.35
    if random.random() < 0.35:
        # l element (-30;0) and r element (0;30) (degrees)
        l_deg = round(random.random() * -30)
        r_deg = round(random.random() * 30)

        counts = augment_image(dual_channel, aug_df, "shear", i, counts,
                               leaf_path, mask_path, shear_img, l=l_deg,
                               r=r_deg)

    return aug_df, counts


# *---------------------------- package __main__ -----------------------------*
def augment_dataset(lseq: LeafSequence, mseq: MaskSequence, **kwargs):
    """

        :param lseq:
        :param mseq:
        :return:
        """
    # linked based on number:  <name>_<image_number>_<tile_number>
    lseq.link_sequences(mseq)

    # dataframe with the possible transformations as columns
    aug_df = pd.DataFrame(index=range(len(lseq.image_objects)),
                          columns=["leaf", "mask", "flip", "translate", "zoom",
                                   "crop", "rotate", "shear"])

    # setting random seed again to be sure
    random.seed(3141)

    # create augmented folders
    base_path = Path(*list(Path(lseq.image_objects[0].path).parts)[:-3])
    create_subfolders(base_path, "augmented")

    # counts of augmented images accepted and rejected
    counts = [0, 0]

    with tqdm(total=len(lseq.image_objects), file=sys.stdout) as pbar:
        for i, leaf in enumerate(lseq.image_objects):
            # create the dual channel image, where leaf is channel 0 and mask
            # is channel 1
            mask = leaf.link

            leaf_path = Path(leaf.path)
            mask_path = Path(mask.path)

            # checking links using numbers explicitly (requires
            # <name>_<image_number>_<tile_number> naming format
            assert (leaf_path.parts[-1].rsplit(".")[0].rsplit("_", 2)[-2:] ==
                    mask_path.parts[-1].rsplit(".")[0].rsplit("_", 2)[-2:]), \
                (f"leaf: {leaf_path} is incorrectly matched with mask:"
                 f" {mask_path}; please check this")

            aug_df["leaf"][i] = leaf_path
            aug_df["mask"][i] = mask_path

            leaf.load_image(**kwargs)
            mask.load_image()

            # Should load image array to keep loading consistent
            dual_channel = stack_images(leaf.image_array, mask.image_array)

            # save RAM
            leaf.unload_extracted_images()
            mask.unload_extracted_images()

            aug_df, counts = augmentation_algorithm(dual_channel, aug_df, i,
                                                    leaf_path, mask_path,
                                                    counts)
            pbar.update(1)

    aug_df.to_csv(base_path.joinpath("augmented", "augmentation_details.csv"))

    LOGGER.info(f"Added {counts[0]} images and rejected {counts[1]} images")
