import random
import shutil
from typing import Union

import imgaug as ia
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split

from src.data_model.data_model import *
from src.helpers.utilities import create_subfolders

random.seed(3141)


# *============================= create dataset ==============================*
def create_dataset_structure(base_dir: Union[Path, str]) -> None:
    """
    Creates a skeleton dataset structure. Train, val, and test folders,
    each with embolism and no-embolism folders are created. A not-used
    folder for downsampled images is also created.

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


def move_data(lseq_list: List[LeafSequence],
              mseq_list: List[MaskSequence],
              dest_root_path: Union[Path, str],
              dest_folder: str = "train") -> List[str]:
    """
    Populates the train folder in the dataset folder, where the dataset 
    folder and its constituents were created using the create_dataset_structure
    function of this module.


    :param lseq_list: list of LeafSequence objects
    :param mseq_list: list of MaskSequence objects
    :param dest_root_path: destination root path; this can either be a Path
     object or a string
    :param dest_folder: destination folder; this is a folder in the
     destination root path
    :return: None

    .. note:: This function requires both leaves and masks to be in the same
              root directory
    """
    if not isinstance(dest_root_path, Path):
        dest_root_path = Path(dest_root_path)

    for lseq, mseq in zip(lseq_list, mseq_list):
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
                dest_root_path.joinpath(dest_folder, "no-embolism", "masks",
                                        x)))

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

        lseq.unload_extracted_images()
        mseq.unload_extracted_images()

    # Note: All leaf and mask tiles must have the same file extension
    # Get the extension using the filenames of the chips of the last chip
    # paths from the above loop
    mask_file_ext = "*." + str(mask_chip_path.parts[-1]).rsplit(".")[1]
    leaf_file_ext = "*." + str(leaf_chip_path.parts[-1]).rsplit(".")[1]

    return [leaf_file_ext, mask_file_ext]


def downsample_dataset(dataset_root_path: Union[Path, str],
                       filename_patterns: List[str],
                       non_embolism_size: float = 0.5) -> \
        Tuple[List[List[str]], List[List[str]]]:
    """
    Downsamples a dataset, where the dataset was created using the
    create_dataset_structure and move_data functions.

    :param dataset_root_path: the root path of the dataset to downsample
    :param filename_patterns: the filename patterns of the both the leaves
     and masks; this list has two elements
    :param non_embolism_size: the size of the no-embolism samples to keep
    :return: two lists, the first has as elements a list of the embolism
     leaves and a list of the embolism masks, and the second as elements a
     list of the chosen no-embolism leaves and a list of the chosen
     no-embolism masks
    """
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

    total_ne_images = len(ignored_masks + chosen_masks)
    percent_moved = len(ignored_leaves) / total_ne_images
    LOGGER.info(f"Downsampled by {len(ignored_leaves)} "
                f"({round(percent_moved * 100)})% non-embolism images")
    LOGGER.info(f"Ratio of embolism to non-embolism leaves has changed from "
                f"1:{total_ne_images / len(e_masks)} to "
                f"1:{len(chosen_masks) / len(e_masks)}")

    return [e_leaves, e_masks], [chosen_leaves, chosen_masks]


def split_dataset(dataset_root_path: Union[Path, str],
                  embolism_objects: List[List[str]],
                  non_embolism_objects: List[List[str]],
                  test_split: float = 0.2,
                  val_split: float = 0.2) -> None:
    """
    Splits a dataset into train, val, and test, by moving a portion of the
    train samples to val and test. The inputs for embolism objects and
    non-embolism objects are usually the outputs returned from the
    downsample_dataset function.

    :param dataset_root_path: the root path of the dataset to split
    :param embolism_objects: a list containing paths to embolism masks and
     leaves; list of leaves at item 0 and list of  masks at item 1
    :param non_embolism_objects:  list containing paths to non-embolism masks
     and leaves; list of leaves at item 0 and list of masks at item 1
    :param test_split: the percentage of the sample to use for the test set
    :param val_split: the percentage of the remaining sample,
     after the test set has been removed, to use for the validation set
    :return: None
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
        f"({round((train_size / total_size) * 100)}%)"
        f"\nValidation set size :  {val_size} "
        f"({round((val_size / total_size) * 100)}%) "
        f"\nTest set size       :  {test_size} "
        f"({round((test_size / total_size) * 100)}%)")


# *---------------------------- package __main__ -----------------------------*
def extract_dataset(lseq_list: List[LeafSequence],
                    mseq_list: List[MaskSequence],
                    dataset_path: Union[Path, str],
                    downsample_split: float,
                    test_split: float,
                    val_split: float,
                    lolo: int = None) -> None:
    """
    Creates a dataset using a list of LeafSequence and MaskSequence objects

    :param lseq_list: a list of LeafSequence objects
    :param mseq_list: a list of MaskSequence objects
    :param dataset_path: the root path of where the dataset should be created
    :param downsample_split: the percentage to no-embolism samples to keep
    :param test_split: the percentage of the sample to use for the test set
    :param val_split: the percentage of the remaining sample,
     after the test set has been removed, to use for the validation set
    :param lolo: the index of the leaf to leave out to use for testing,
     if a complete leaf should be used for testing; the index corresponds to
     the leafs position in the lseq_list and mseq_list
    :return: None
    """
    # will create a structure iff one does not exist in the correct
    # format at the specified path
    create_dataset_structure(dataset_path)

    if isinstance(lolo, int):
        # isolate the leaf to leave out
        lseq_lolo = [lseq_list.pop(lolo)]
        mseq_lolo = [mseq_list.pop(lolo)]

        _ = move_data(lseq_lolo, mseq_lolo, dataset_path, "test")

    filename_patterns = move_data(lseq_list, mseq_list, dataset_path)

    # non_emb_list will contain the filenames of chosen non-embolism images
    emb_list, non_emb_list = downsample_dataset(dataset_path,
                                                filename_patterns,
                                                downsample_split)

    split_dataset(dataset_path, emb_list, non_emb_list, test_split, val_split)


# *============================= augment dataset =============================*
# *----------------------------- transformations -----------------------------*
def flip_flop(leaf_image_array: np.array,
              mask_segmap: ia.augmentables.segmaps.SegmentationMapsOnImage,
              orientation: str,
              seed: int = 3141) -> \
        Tuple[np.array, ia.augmentables.segmaps.SegmentationMapsOnImage]:
    """
    Reflects a sample on either on the x or y-axis

    :param leaf_image_array: the input image
    :param mask_segmap: the mask segmentation map
    :param orientation: whether to flip horizontally or vertically
    :param seed: the random seed
    :return: updated leaf input and mask
    """
    if orientation == "horizontal":
        flip_hr = iaa.Fliplr(seed=seed)
        flipped_images = flip_hr.augment_image(leaf_image_array)
        mask_segmap = flip_hr.augment_segmentation_maps(mask_segmap)
    elif orientation == "vertical":
        flip_vr = iaa.Flipud(seed=seed)
        flipped_images = flip_vr.augment_image(leaf_image_array)
        mask_segmap = flip_vr.augment_segmentation_maps(mask_segmap)
    else:
        raise ValueError("please provide either 'horizontal' or 'vertical as "
                         "the orientation'")

    return flipped_images, mask_segmap


def translate_img(leaf_image_array: np.array,
                  mask_segmap: ia.augmentables.segmaps.SegmentationMapsOnImage,
                  x: float,
                  y: float,
                  seed: int = 3141) -> \
        Tuple[np.array, ia.augmentables.segmaps.SegmentationMapsOnImage]:
    """
    Translates an image. The padding pixels are black.

    :param leaf_image_array: the input image
    :param mask_segmap: the mask segmentation map
    :param x: percentage to shift on the x-axis (between -1 and 1)
    :param y: percentage to shift on the y-axis (between -1 and 1)
    :param seed: the random seed
    :return: updated leaf input and mask
    """
    rotate = iaa.Affine(translate_percent=(x, y), seed=seed)
    leaf_image = rotate.augment_image(leaf_image_array)
    mask_segmap = rotate.augment_segmentation_maps(mask_segmap)

    return leaf_image, mask_segmap


def rotate_img(leaf_image_array: np.array,
               mask_segmap: ia.augmentables.segmaps.SegmentationMapsOnImage,
               l: float,
               r: float,
               seed: int = 3141) -> \
        Tuple[np.array, ia.augmentables.segmaps.SegmentationMapsOnImage]:
    """
    Rotates an image a random amount of degrees between (l,r). The padding
    pixels are black.

    :param leaf_image_array: the input image
    :param mask_segmap: the mask segmentation map
    :param l: degrees to rotate to the left
    :param r: degrees to rotate to the right
    :param seed: the random seed
    :return: updated leaf input and mask
    """
    rotate = iaa.Affine(rotate=(l, r), seed=seed)
    leaf_image = rotate.augment_image(leaf_image_array)
    mask_segmap = rotate.augment_segmentation_maps(mask_segmap)

    return leaf_image, mask_segmap


def shear_img(leaf_image_array: np.array,
              mask_segmap: ia.augmentables.segmaps.SegmentationMapsOnImage,
              l: float,
              r: float,
              seed: int = 3141) -> \
        Tuple[np.array, ia.augmentables.segmaps.SegmentationMapsOnImage]:
    """
    Shears an image a random amount of degrees between (l,r). The padding
    pixels are black.

    :param leaf_image_array: the input image
    :param mask_segmap: the mask segmentation map
    :param l: degrees to shear to the left
    :param r: degrees to shear to the right
    :param seed: the random seed
    :return: updated leaf input and mask
    """
    # Shear in degrees
    shear = iaa.Affine(shear=(l, r), seed=seed)

    leaf_image = shear.augment_image(leaf_image_array)
    mask_segmap = shear.augment_segmentation_maps(mask_segmap)

    return leaf_image, mask_segmap


def crop_img(leaf_image_array: np.array,
             mask_segmap: ia.augmentables.segmaps.SegmentationMapsOnImage,
             v: float,
             h: float,
             seed: int = 3141) -> \
        Tuple[np.array, ia.augmentables.segmaps.SegmentationMapsOnImage]:

    """
    Crops an image. The padding pixels are black.

    :param leaf_image_array: the input image
    :param mask_segmap: the mask segmentation map
    :param v: the percent to crop vertically
    :param h: the percent to crop horizontally
    :param seed: the random seed
    :return: updated leaf input and mask
    """
    crop = iaa.Crop(percent=(v, h), seed=seed)
    leaf_image = crop.augment_image(leaf_image_array)
    mask_segmap = crop.augment_segmentation_maps(mask_segmap)

    return leaf_image, mask_segmap


def zoom_in_out(leaf_image_array: np.array,
                mask_segmap: ia.augmentables.segmaps.SegmentationMapsOnImage,
                x: float,
                y: float,
                seed: int = 3141) ->  \
        Tuple[np.array, ia.augmentables.segmaps.SegmentationMapsOnImage]:
    """
    Zooms in or out of an image. The padding pixels are black.

    :param leaf_image_array: the input image
    :param mask_segmap: the mask segmentation map
    :param x: % to zoom on the x-axis; 1 is 100%
    :param y: % to zoom on the x-axis; 1 is 100%
    :param seed: the random seed
    :return: updated leaf input and mask
    """
    scale_im = iaa.Affine(scale={"x": x, "y": y}, seed=seed)
    leaf_image = scale_im.augment_image(leaf_image_array)
    mask_segmap = scale_im.augment_segmentation_maps(mask_segmap)

    return leaf_image, mask_segmap


# *--------------------------------- helpers ---------------------------------*
def save_image(leaf: Leaf, mask: Mask, aug_type: str) -> None:
    """
    Saves an augmented Leaf and Mask. The new filename includes the details
    of the augmentation.

    :param leaf: A Leaf object, with augmented image
    :param mask: A Mask object, with augmented image
    :param aug_type: the details of the augmentation to be added to the new
     filename
    :return: None
    """
    old_paths = [leaf.path, mask.path]
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
    cv2.imwrite(str(new_paths[0]), leaf.image_array)
    cv2.imwrite(str(new_paths[1]), mask.image_array.astype(np.uint8))


def augment_image(leaf: np.array,
                  mask: np.array,
                  df: pd.DataFrame,
                  aug_type: str,
                  index: int,
                  counts: List[int],
                  func, **kwargs) -> List[int]:
    """
    Applies an augmentation to a sample. The augmented sample is rejected if
    the augmentation removes all embolisms from the image. If the
    augmentation is accepted, it is saved, and the aug_df is updated with
    the details of the augmentation. The updates to the df are made in
    place, so the df is mutated despite not being returned.

    :param leaf: the input leaf
    :param mask: the input mask
    :param df: the augmentation df
    :param aug_type: the type of augmentation
    :param index: the index of the sample in the input df
    :param counts: the counts of augmentation acceptance and rejection; the
     list has two elements
    :param func: the augmentation function
    :param kwargs: the kwargs for the augmentation function
    :return: updated counts
    """
    segmap = ia.augmentables.segmaps.SegmentationMapsOnImage(
        mask.image_array, mask.image_array.shape)
    leaf.image_array, segmap = func(leaf.image_array, segmap, **kwargs)
    mask.image_array = segmap.get_arr()
    # only save an image if it has an embolism
    # binary segmentation problem so we know that if there are two pixel
    # intensities there are embolisms

    if len(np.unique(mask.image_array)) > 1:
        save_image(leaf, mask, aug_type)

        df[aug_type][index] = ', '.join(
            [f'{k}: {v}' for k, v in kwargs.items()])

        counts[0] += 1
    else:
        counts[1] += 1

    return counts


def augmentation_algorithm(leaf: np.array,
                           mask: np.array,
                           aug_df: pd.DataFrame,
                           i: int,
                           counts: List[int]) -> \
        Tuple[pd.DataFrame, List[int]]:
    """
    Passes the sample through a series of possible augmentations: flip_flop,
    translate, zoom, crop, rotate, and shear. These augmentations are each
    applied with probability of 0.5. The augmented images are saved. The input
    DataFrame is updated with augmentations that were applied to the image.
    The count of augmentations is also updated.

    :param leaf: the leaf to augment
    :param mask: the mask to augment
    :param aug_df: the augmentation df
    :param i: the position in the dataframe corresponding to the sample
    :param counts: a list of counts, the first number is a count of times an
     augmentation was accepted and the second is the count of times an
     augmentation was rejected.
    :return: None
    """
    # P(flip) = 0.5
    if random.random() < 0.5:
        # P(H | flip) = 0.5 | P(V | flip) = 0.5
        if random.random() < 0.5:
            orientation = "horizontal"
        else:
            orientation = "vertical"

        counts = augment_image(leaf, mask, aug_df, "flip", i, counts,
                               flip_flop, orientation=orientation)

    # P(translate) = 0.5
    if random.random() < 0.5:
        # zoom in and out between -25% and 25%
        x_per = round(random.uniform(-0.25, 0.25), 2)
        y_per = round(random.uniform(-0.25, 0.25), 2)

        counts = augment_image(leaf, mask, aug_df, "translate", i, counts,
                               translate_img, x=x_per, y=y_per)

    # P(zoom) = 0.5
    if random.random() < 0.5:
        # zoom in and out between 150% and 50%
        x_per = round(random.uniform(1.5, 0.5), 2)
        y_per = round(random.uniform(1.5, 0.5), 2)

        counts = augment_image(leaf, mask, aug_df, "zoom", i, counts,
                               zoom_in_out, x=x_per, y=y_per)

    # P(crop) = 0.5
    if random.random() < 0.5:
        # crop between 5% and 30% of the image
        v_per = round(random.uniform(0.05, 0.3), 2)
        h_per = round(random.uniform(0.05, 0.3), 2)

        counts = augment_image(leaf, mask, aug_df, "crop", i, counts,
                               crop_img, v=v_per, h=h_per)

    # P(rotate) = 0.5
    if random.random() < 0.5:
        # l element (-90;0) and r element (0;90) (degrees)
        l_deg = round(random.random() * -90)
        r_deg = round(random.random() * 90)

        counts = augment_image(leaf, mask, aug_df, "rotate", i, counts,
                               rotate_img, l=l_deg, r=r_deg)

    # P(sheer) = 0.5
    if random.random() < 0.5:
        # l element (-30;0) and r element (0;30) (degrees)
        l_deg = round(random.random() * -30)
        r_deg = round(random.random() * 30)

        counts = augment_image(leaf, mask, aug_df, "shear", i, counts,
                               shear_img, l=l_deg, r=r_deg)

    return aug_df, counts


# *---------------------------- package __main__ -----------------------------*
def augment_dataset(lseq: LeafSequence, mseq: MaskSequence, **kwargs) -> None:
    """
    Augments a dataset using the provided LeafSequence and MaskSequence.
    Both the LeafSequence and MaskSequence are usually created using the
    train folder from the dataset. The augmented files are saved in a folder
    called augmented at the common root folder of the leaf and mask
    sequence. A csv with the details of augmentation is also saved.

    :param lseq: LeafSequence object of the dataset
    :param mseq: MaskSequence object of the dataset
    :return: None
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
            assert (leaf_path.parts[-1].rsplit(".")[0].rsplit("_", 2)[-1:] ==
                    mask_path.parts[-1].rsplit(".")[0].rsplit("_", 2)[-1:]), \
                (f"leaf: {leaf_path} is incorrectly matched with mask:"
                 f" {mask_path}; please check this")

            aug_df["leaf"][i] = leaf_path
            aug_df["mask"][i] = mask_path

            leaf.load_image(**kwargs)
            mask.load_image()

            # save RAM
            leaf.unload_extracted_images()
            mask.unload_extracted_images()

            aug_df, counts = augmentation_algorithm(
                leaf, mask, aug_df, i, counts)
            pbar.update(1)

    aug_df.to_csv(base_path.joinpath("augmented", "augmentation_details.csv"))

    LOGGER.info(f"Added {counts[0]} images and rejected {counts[1]} images")
