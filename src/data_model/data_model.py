import logging
import os
import sys
from abc import ABC, abstractmethod
from glob import glob
from itertools import chain
from math import log10, floor, ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import PIL.Image
import PIL.ImageSequence
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageChops
from tqdm import tqdm

from src.actions.plots import plot_embolism_profile
from src.helpers import utilities
from src.helpers.describe_leaf import binarise_image
from src.helpers.extract_dataset import chip_image, pad_chip, chip_range
from src.model.model import Model

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# *================================ Sequences ================================*
# *----------------------------- Abstract Class ------------------------------*
class _ImageSequence(ABC):
    """
    Abstract image sequence class
    """

    def __init__(self,
                 folder_path: str = None,
                 filename_pattern: str = None,
                 file_list: List[str] = None,
                 creation_mode: bool = False):
        """
        Instantiates an image sequence object.

        :param folder_path: path that contains the image sequence
        :param filename_pattern: the filename pattern of image files
        :param file_list: a list of filenames, this can be used instead of
         providing a folder path and filename pattern
        :param creation_mode: whether the image sequence object should be
         instantiated in creation mode
        """
        if file_list is not None:
            self.file_list = file_list

            self.num_files = len(self.file_list)

            if self.num_files == 0:
                LOGGER.debug("The file list is empty")

        elif folder_path is not None and filename_pattern is not None:
            self.folder_path = folder_path
            self.filename_pattern = filename_pattern

            self.file_list = sorted([f for f in glob(
                self.folder_path + self.filename_pattern, recursive=True)])

            self.num_files = len(self.file_list)

        else:
            self.num_files = 0

        # two modes - original mode | extracted mode:
        self.creation_mode = creation_mode

        self.image_objects = []
        self.link = None

        # EDA objects
        self.unique_range = np.array([])

        self.intersection_list = []
        self.unique_range_list = []
        self.embolism_percent_list = []
        self.linked_path_list = []
        self.has_embolism_list = []

    # *______________________ loading | linking Images _______________________*
    # abstract due to signature mismatch in child classes
    @abstractmethod
    def load_extracted_images(self,
                              ImageClass,
                              load_image: bool = False,
                              disable_pb: bool = False,
                              **kwargs) -> None:
        """
        Instantiates objects of class ImageClass using the file_list
        attribute  and appends to image_objects.

        :param ImageClass: the image class, either Leaf or Mask
        :param load_image: whether to load the image array belonging to
         ImageClass being created
        :param disable_pb: whether the progress bar should be disabled
        :param kwargs: kwargs for loading the image; applies if load_image is
         true
        :return:

        .. note:: This function is intended to load extracted images and
         will only work if the image is not instantiated in creation_mode.
        """
        if self.creation_mode:
            raise Exception("The file list contains original images, "
                            "not extracted images. This function is not "
                            "applicable...")

        LOGGER.debug("Erasing existing image objects. If sequences are "
                     "linked please relink them...")
        self.image_objects = []
        with tqdm(total=len(self.file_list), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for i, filename in enumerate(self.file_list):
                self.image_objects.append(ImageClass(filename))
                if load_image:
                    self.image_objects[i].load_image(**kwargs)
                pbar.update(1)

    def load_image_array(self, disable_pb: bool = False, **kwargs) -> None:
        """
        Loads all image arrays belonging to the Image objects in the sequence.

        :param disable_pb: whether the progress bar should be disabled
        :param kwargs: kwargs for loading the image
        :return: None
        """
        with tqdm(total=len(self.file_list), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for image in self.image_objects:
                image.load_image(**kwargs)
                pbar.update(1)

    def link_sequences(self,
                       SequenceObject,
                       sort_by_filename: bool = True) -> None:
        """
        Links a sequence object to another sequence object of a different
        type; i.e. links a MaskSequence to LeafSequence and vice versa. The
        link is made using the individual ImageClass objects.

        :param SequenceObject: the sequence object to link
        :param sort_by_filename: whether the image objects should be sorted
         by filename before linking; using this parameter assumes that the
         leaf and mask files to be linked are in the correct order once sorted
        :return: None
        """
        self.link = SequenceObject
        image_sequence = SequenceObject.image_objects

        index_self = list(range(len(self.image_objects)))
        index_input = list(range(len(image_sequence)))

        if sort_by_filename:
            path_list_self_sequence = [image.path for image in
                                       self.image_objects]
            path_list_input_sequence = [image.path for image in image_sequence]

            # if the input is an original file folder then num_files will be
            # incorrect => len(self.image_objects)
            # don't want to mutate the existing object
            index_self = [i for _, i in
                          sorted(zip(path_list_self_sequence, index_self),
                                 key=lambda pair: pair[0])]
            index_input = [i for _, i in
                           sorted(zip(path_list_input_sequence, index_input),
                                  key=lambda pair: pair[0])]

        for i, j in zip(index_self, index_input):
            # Do we need a two-way link?
            self.image_objects[i].link_me(image_sequence[j])
            image_sequence[j].link_me(self.image_objects[i])

    # *___________________________ pre-processing ____________________________*
    def trim_image_sequence(self, x_size_dir: Optional[Tuple[int, int]] = None,
                            y_size_dir: Optional[Tuple[int, int]] = None,
                            overwrite: bool = True, disable_pb=False) -> None:
        """
        Trims all images in an image sequence.

        :param disable_pb: whether the progress bar should be disabled
        :param y_size_dir: a tuple of (output size, trim_direction), where
         trim direction is either 1 or -1, which indicates to trim from
         either the top or bottom respectively
        :param x_size_dir: a tuple of (output size, trim_direction), where
         trim direction is either 1 or -1, which indicates to trim from
         either the left or right respectively
        :param overwrite: whether images that exist at the same file path
         should be overwritten
        :return: None

        .. Note:: Parameters apply to all images in the image sequence
        """
        # TODO: Implement option to use either x or y mode for output size
        with tqdm(total=len(self.image_objects), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for image in self.image_objects:
                if x_size_dir and y_size_dir:
                    if image.image_array.shape != (
                            y_size_dir[0], x_size_dir[0]):
                        image.trim_image(x_size_dir, y_size_dir, overwrite)
                elif y_size_dir:
                    if image.image_array.shape[0] != y_size_dir[0]:
                        image.trim_image(y_size_dir=y_size_dir,
                                         overwrite=overwrite)
                elif x_size_dir:
                    if image.image_array.shape[1] != x_size_dir[0]:
                        image.trim_image(x_size_dir=x_size_dir,
                                         overwrite=overwrite)
                pbar.update(1)

    # *_________________________________ EDA _________________________________*
    def get_unique_range_list(self, disable_pb: bool = False) -> None:
        """
        Updates the unique_range_list attribute, which is a list of the
        unique pixel intensities per image.

        :param disable_pb: whether the progress bar should be disabled
        :return: None
        """
        self.unique_range_list = []

        with tqdm(total=len(self.image_objects), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for image in self.image_objects:
                # no option to overwrite ...
                if image.unique_range.size == 0:
                    image.extract_unique_range()

                    self.unique_range_list.append(image.unique_range)

                pbar.update(1)

    def get_unique_sequence_range(self) -> None:
        """
        Updates the unique_range attribute, which is the unique pixel
        intensities over an entire image sequence.

        :return: None
        """
        if not self.unique_range_list:
            self.get_unique_range_list()

        self.unique_range = np.unique(list(chain.from_iterable(
            self.unique_range_list)))

    def get_intersection_list(self, disable_pb: bool = False) -> None:
        """
        Updates intersection_list attribute, which is a list of the
        intersection between each image and all images before that image in
        the sequence.

        :param disable_pb: whether the progress bar should be disabled
        :return: None
        """
        self.intersection_list = []

        image_shape = self.image_objects[0].image_array.shape
        # assumes images are the same size
        combined_image = np.empty(shape=(image_shape[0], image_shape[1]),
                                  dtype='object')

        with tqdm(total=len(self.image_objects), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for image in self.image_objects:
                combined_image = image.extract_intersection(combined_image)
                self.intersection_list.append(image.intersection)
                pbar.update(1)

    def get_embolism_percent_list(self, disable_pb: bool = False):
        """
        Updates embolism_percent_list attribute, which is a list of the % of
        embolisms in each image in the sequence.


        :param disable_pb: whether the progress bar should be disabled
        :return: None
        """
        self.embolism_percent_list = []

        with tqdm(total=len(self.image_objects), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for image in self.image_objects:
                if image.embolism_percent is None:
                    image.extract_embolism_percent()

                self.embolism_percent_list.append(image.embolism_percent)
                pbar.update(1)

    def get_has_embolism_list(self, disable_pb: bool = False):
        """
        Updates has_embolism_list attribute, which is a list of the booleans
        indicating whether an image in a sequence has an embolism or not.


        :param disable_pb: whether the progress bar should be disabled
        :return: None
        """
        self.has_embolism_list = []

        with tqdm(total=len(self.image_objects), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for image in self.image_objects:
                if image.has_embolism is None:
                    image.extract_has_embolism()

                self.has_embolism_list.append(image.has_embolism)
                pbar.update(1)

    # *______________________________ utilities ______________________________*
    def get_eda_dataframe(self,
                          options: Dict,
                          csv_name: str = None,
                          disable_pb: bool = False) -> pd.DataFrame:
        """
        Creates an EDA DataFrame using the iamges in the sequence. If a csv
        name is provided the DataFrame is saved.

        :param options: the options of what to include in the DF; the
         options are: linked filename, unique range, embolism_percent,
         intersection, and has_embolism
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be saved
        :param disable_pb: whether the progress bar should be disabled
        :return: EDA DF
        """
        output_dict = {"names": list(map(
            lambda image: image.path.rsplit("/", 1)[1], self.image_objects))}

        if options["linked_filename"]:
            # assumes files have been linked
            output_dict["links"] = []
            for image in self.image_objects:
                if image.link is not None:
                    output_dict["links"].append(
                        image.link.path.rsplit("/", 1)[1])
                else:
                    output_dict["links"].append("")

        # Unique range
        if options["unique_range"]:
            if not self.unique_range_list:
                self.get_unique_range_list(disable_pb)

            output_dict["unique_range"] = self.unique_range_list

        # Embolism percentages
        if options["embolism_percent"]:
            if not self.embolism_percent_list:
                self.get_embolism_percent_list(disable_pb)

            output_dict["embolism_percent"] = self.embolism_percent_list

        # Extracting intersections
        if options["intersection"]:
            if not self.intersection_list:
                self.get_intersection_list(disable_pb)

            output_dict["intersection"] = self.intersection_list

            # Extracting intersections
        if options["has_embolism"]:
            if not self.has_embolism_list:
                self.get_has_embolism_list(disable_pb)

            output_dict["has_embolism"] = self.has_embolism_list

        output_df = pd.DataFrame(output_dict)
        # Saving the results
        if csv_name:
            output_df.to_csv(csv_name)

        return output_df

    @abstractmethod
    def get_databunch_dataframe(self,
                                lseq,
                                mseq,
                                embolism_only: bool = False,
                                csv_name: str = None) -> \
            Tuple[pd.DataFrame, str]:
        """
        Extracts a databunch dataframe using the images in this sequence. The
        first field is the leaf path and the second field is the mask name.
        This is useful for Fastai. If a csv name is provided the DataFrame
        is saved.

        :param lseq: a LeafSequence object
        :param mseq: a MaskSequence object
        :param embolism_only: whether only leaves with embolisms should be used
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: DataBunch DF and sequence root folder path
        """

        output_dict = {"leaf_name":
                           list(map(lambda image: image.path.rsplit("/", 1)[1],
                                    lseq.image_objects)),
                       "mask_path": []}

        for image in lseq.image_objects:
            if image.link is not None:
                output_dict["mask_path"].append(image.link.path)
            else:
                output_dict["mask_path"].append("")

        output_df = pd.DataFrame(output_dict)
        folder_path = lseq.image_objects[0].path.rsplit("/", 1)[0]

        if embolism_only:
            # it's possible for there to be more mask sequence objects than
            # leaf sequence - mseq.has_embolism would fail in these cases
            linked_has_embolism_list = [
                image.link.has_embolism for image in lseq.image_objects]
            # only considers images with a corresponding mask which has an
            # embolism
            output_df = output_df[linked_has_embolism_list]

        # Saving the results
        if csv_name:
            output_df.to_csv(csv_name)

        return output_df, folder_path

    def unload_extracted_images(self):
        for image in self.image_objects:
            image.image_array = None

    def sort_image_objects_by_filename(self):
        self.image_objects = sorted(self.image_objects,
                                    key=lambda image: image.path)

    def plot_profile(self,
                     show: bool = True,
                     output_path: str = None,
                     leaf_name: str = None,
                     **kwargs):
        """
        Plots an embolism profile for the sequence

        :param show: whether the plot should be shown; if the plot is
         being saved, the user may not want to display the plot
        :param output_path: path to save the plot, if None, the plot will not be
         saved
        :param leaf_name: the name of the leaf to use in the title
        :param kwargs: subplot kwargs
        :return:
        """
        plot_embolism_profile(
            embolism_percentages=self.embolism_percent_list,
            intersections=self.intersection_list, leaf_name=leaf_name,
            output_path=output_path, show=show, **kwargs)


# *---------------------------------- Mixin ----------------------------------*
class _CurveSequenceMixin:
    """
    Adds functions to allow curve sequences to operate at a tile level
    """

    # *__________________ tiling & loading | linking Tiles ___________________*
    def tile_sequence(self,
                      length_x: int,
                      stride_x: int,
                      length_y: int,
                      stride_y: int,
                      output_path: str = None,
                      overwrite: bool = False,
                      memory_saving: bool = True) -> None:
        """
        Tiles the images in the Image objects in the sequence.

        :param length_x: the x-length of the tile
        :param stride_x: the size of the x stride
        :param length_y: the y-length of the tile
        :param stride_y: the size of the y stride
        :param output_path: output path of where the tiles should be saved;
         if no path is  provided, tiles are saved in a default location
        :param overwrite: whether tiles that exist at the same file path should
         be overwritten
        :param memory_saving: whether the tiles should be unloaded from the
         their parent Image objects once they have been created
        :return: None
        """
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                image.tile_me(length_x, stride_x, length_y, stride_y,
                              output_path, overwrite)
                if memory_saving:
                    image.unload_extracted_images()
                pbar.update(1)

    def load_tile_sequence(self,
                           load_image: bool = False,
                           folder_path: str = None,
                           filename_pattern: str = None,
                           **kwargs) -> None:
        """
        Loads all tile objects belonging to the Image objects in the sequence.

        :param load_image: whether the tile arrays should also be loaded
        :param folder_path: the folder path of the tiles
        :param filename_pattern: the filename pattern of the tiles
        :param kwargs: kwargs for loading the image; applies if load_image is
         true
        :return: None
        """
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                image.load_tile_paths(folder_path, filename_pattern)
                image.load_extracted_images(load_image, disable_pb=True,
                                            **kwargs)
                pbar.update(1)

    def link_tiles(self, sort_by_filename: bool = True) -> None:
        """
        Links all tiles to the tiles belonging to the Image object's link.
        This requires the images to first be linked.

        :param sort_by_filename: whether the tile objects should be sorted
         by filename before linking; using this parameter assumes that the
         leaf and mask tiles to be linked are in the correct order once sorted
        :return: None
        """
        # Requires images to be linked
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                if image.link is None:
                    LOGGER.debug(f"Image {image.path} was not linked")
                    continue

                # overwrites link with itself
                image.link_sequences(image.link, sort_by_filename)
                pbar.update(1)

    def get_tile_databunch_df(self,
                              lseq,
                              mseq,
                              tile_embolism_only: bool = False,
                              leaf_embolism_only: bool = False,
                              csv_name: str = None) -> \
            Tuple[pd.DataFrame, List[str]]:
        """
        Extracts a combined databunch df using all tiles belonging to the
        Image objects in the sequence.  The first field is the leaf tile path
        and the second field is the mask tile name. This is useful for Fastai.
        If a csv name is provided the DataFrame is saved.

        :param lseq: a LeafSequence object
        :param mseq: a MaskSequence object
        :param tile_embolism_only: whether only tiles with embolisms should be
         used
        :param leaf_embolism_only: whether only leaves with embolisms should be
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: combined DataBunch DF and list of image root folder path
        """
        databunch_df_list = []
        folder_list = []

        lseq.load_tile_sequence()
        mseq.load_tile_sequence()

        lseq.link_tiles()

        if leaf_embolism_only:
            LOGGER.debug("Extracting has_embolism_list for full masks")
            mseq.load_image_array()
            mseq.get_embolism_percent_list()
            mseq.get_has_embolism_list()

        LOGGER.debug("Creating Tile Datbunch csv")

        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in lseq.image_objects:
                if leaf_embolism_only:
                    # Assumes linked sequences have been provided.
                    if not image.link.has_embolism:
                        # Skip this image if the mask has no embolism
                        continue

                if tile_embolism_only:
                    mask_image = image.link
                    mask_image.load_image_array(disable_pb=True)
                    mask_image.get_embolism_percent_list(disable_pb=True)
                    mask_image.get_has_embolism_list(disable_pb=True)

                    # To save memory
                    mask_image.unload_extracted_images()

                df, folder_path = image.get_databunch_dataframe(
                    tile_embolism_only)
                folder_list.append(folder_path)

                df["folder_path"] = folder_path
                databunch_df_list.append(df)

                pbar.update(1)

        full_databunch_df = pd.concat(databunch_df_list)

        if csv_name:
            full_databunch_df.to_csv(csv_name)

        return full_databunch_df, folder_list

    def get_tile_eda_df(self, options, csv_name: str = None):
        """
        Creates an EDA DataFrame using all tiles belonging to the
        Image objects in the sequence If a csv name is provided the
        DataFrame is saved.

        :param options: the options of what to include in the DF; the
         options are: linked filename, unique range, embolism_percent,
         intersection, and has_embolism
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be saved
        :return: EDA DF
        """
        self.load_tile_sequence()
        self.link.load_tile_sequence()

        # Requires sequence to be linked
        if options["linked_filename"]:
            self.link_tiles()

        eda_df_list = []
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for i, image in enumerate(self.image_objects):
                image.load_image_array(disable_pb=True)

                df = image.get_eda_dataframe(options, disable_pb=True)
                eda_df_list.append(df)

                image.unload_extracted_images()

                pbar.update(1)

        full_eda_df = pd.concat(eda_df_list)

        if csv_name:
            full_eda_df.to_csv(csv_name)

        return full_eda_df


# *----------------------------- Implementation ------------------------------*
# *__________________________________ Leaf ___________________________________*
class LeafSequence(_CurveSequenceMixin, _ImageSequence):
    """
    A sequence of full size Leaf Images
    """

    def __init__(self,
                 folder_path: str = None,
                 filename_pattern: str = None,
                 file_list: List[str] = None,
                 creation_mode: bool = False):
        """
        Instantiates a LeafSequence object.

        :param folder_path: path that contains the image sequence
        :param filename_pattern: the filename pattern of image files
        :param file_list: a list of filenames, this can be used instead of
         providing a folder path and filename pattern
        :param creation_mode: whether the image sequence object should be
         instantiated in creation mode; this attribute allows the user to
         determine whether the file list pertains to raw images or
         differenced images
        """
        _ImageSequence.__init__(self, folder_path, filename_pattern,
                                file_list, creation_mode)
        if self.num_files == 0:
            LOGGER.warning("The file list is empty")

    # *____________________________ extraction ______________________________*
    def extract_changed_leaves(self,
                               output_path: str,
                               dif_len: int = 1,
                               overwrite: bool = False,
                               shift_256: bool=False,
                               combination_function=ImageChops.subtract_modulo)\
            -> None:
        """
        Extracts and saves changed leaf images. This uses the filepath
        list created when the leaf sequence is instantiated.

        :param output_path: where the differenced leaves should be saved
        :param dif_len: the step size between the leaves to be differenced
        :param overwrite: whether images that exist at the same file path
         should be overwritten
        :param shift_256: whether images should be shifted by 256; this also
         means that images will saved as uint16
        :param combination_function: the combination function to be used;
         the default is to difference leaves
        :return: None
        """
        output_folder_path, output_file_name = output_path.rsplit("/", 1)
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        if dif_len == 0:
            old_image = self.file_list[0]
            step_size = 1
        else:
            step_size = dif_len

        placeholder_size = floor(log10(self.num_files)) + 1

        with tqdm(total=len(self.file_list) - dif_len,
                  file=sys.stdout) as pbar:
            for i in range(0, self.num_files - step_size):
                if dif_len != 0:
                    old_image = self.file_list[i]

                new_image = self.file_list[i + step_size]

                final_filename = utilities.create_file_name(
                    output_folder_path, output_file_name, i, placeholder_size)

                self.image_objects.append(Leaf(parents=[old_image, new_image],
                                               sequence_parent=self))
                self.image_objects[i].extract_me(
                    final_filename, combination_function, shift_256, overwrite)

                pbar.update(1)

    # *_______________________________ loading _______________________________*
    def load_extracted_images(self,
                              load_image: bool = False,
                              disable_pb: bool = False,
                              shift_256: bool = False,
                              transform_uint8: bool = False) -> None:
        """
        Instantiates Leaf objects using the file_list attribute and appends
        these objects to the image_objects attribute.

        :param load_image: whether to load the image array belonging to
         Leaf being created
        :param disable_pb: whether the progress bar should be disabled
        :param shift_256: whether images should be shifted by 256; applies
         if load_image is true
        :param transform_uint8: whether images transformed to a uint8 format;
         applies if load_image is true
        :return: None
        """
        super().load_extracted_images(Leaf, load_image, disable_pb,
                                      shift_256=shift_256,
                                      transform_uint8=transform_uint8)

    def load_image_array(self,
                         disable_pb: bool = False,
                         shift_256: bool = False,
                         transform_uint8: bool = False) -> None:
        """
        Loads all image arrays belonging to the Leaf objects in the sequence.

        :param disable_pb: whether the progress bar should be disabled
        :param shift_256: whether images should be shifted by 256
        :param transform_uint8: whether images transformed to a uint8 format
        :return: None
        """
        # not strictly necessary but more user friendly
        super().load_image_array(disable_pb, shift_256=shift_256,
                                 transform_uint8=transform_uint8)

    def load_tile_sequence(self,
                           load_image: bool = False,
                           folder_path: str = None,
                           filename_pattern: str = None,
                           shift_256: bool = False,
                           transform_uint8: bool = False) -> None:
        """
        Loads all tile objects belonging to the Leaf objects in the sequence.

        :param load_image: whether the tile arrays should also be loaded
        :param folder_path: the folder path of the tiles
        :param filename_pattern: the filename pattern of the tiles
        :param shift_256: whether images should be shifted by 256; applies
         if load_image is true
        :param transform_uint8: whether images transformed to a uint8 format;
         applies if load_image is true
        :return: None
        """
        super().load_tile_sequence(load_image, folder_path, filename_pattern,
                                   shift_256=shift_256,
                                   transform_uint8=transform_uint8)

    # *_____________________________ prediction ______________________________*
    def predict_leaf_sequence(self, model: Model,
                              x_tile_length: int = None,
                              y_tile_length: int = None,
                              memory_saving: bool = True,
                              overwrite: bool = False,
                              save_prediction: bool = True,
                              shift_256: bool = False,
                              transform_uint8: bool = False,
                              threshold: float = 0.5,
                              **kwargs) -> None:
        """
        Predicts segmentation maps using the Leaves in the sequence. The
        model used should implement a predict tile method. If memory saving
        is set to false a prediction array is assigned to each Leaf object
        in the sequence.

        :param model: a model which inherits Model and hence implements a
         predict tile method
        :param x_tile_length: the x length of the tile used in the original
         training
        :param y_tile_length: the y length of the tile used in the original
         training
        :param memory_saving: if set to True, both the image array and
         prediction array are set to None; this should only be set to true
         if the predictions are being saved
        :param overwrite: whether images that exist at the same file path
         should be overwritten
        :param save_prediction: whether the prediction should be saved
        :param shift_256: whether images should be shifted by 256
        :param transform_uint8: whether images transformed to a uint8 format
        :param threshold: the threshold to use when saving predictions; i.e. a
         pixel is saved as an embolism if p(embolism) > threshold
        :param kwargs: kwargs for the predict tile function
        :return: None
        """
        # if shifted by 256 then apply im1 > im2 post processing
        if shift_256:
            kwargs["post_process"] = True
        else:
            kwargs["post_process"] = False

        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for leaf in self.image_objects:
                leaf.predict_leaf(model, x_tile_length, y_tile_length,
                                  memory_saving, overwrite, save_prediction,
                                  shift_256, transform_uint8, threshold,
                                  **kwargs)
                pbar.update(1)

    # *______________________________ utilities ______________________________*
    def get_databunch_dataframe(self,
                                embolism_only: bool = False,
                                csv_name: str = None) -> \
            Tuple[pd.DataFrame, str]:
        """
        Extracts a databunch dataframe using the images in this sequence. The
        first field is the leaf path and the second field is the mask name.
        This is useful for Fastai. If a csv name is provided the DataFrame
        is saved.

        :param embolism_only: whether only leaves with embolisms should be used
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: DataBunch DF and sequence root folder path
        """
        return super().get_databunch_dataframe(lseq=self, mseq=self.link,
                                               embolism_only=embolism_only,
                                               csv_name=csv_name)

    def get_tile_databunch_df(self, mseq,
                              tile_embolism_only: bool = False,
                              leaf_embolism_only: bool = False,
                              csv_name: str = None) -> \
            Tuple[pd.DataFrame, List[str]]:
        """
        Extracts a combined databunch df using all tiles belonging to the
        Image objects in the sequence.  The first field is the leaf tile path
        and the second field is the mask tile name. This is useful for Fastai.
        If a csv name is provided the DataFrame is saved.

        :param mseq: a MaskSequence object
        :param tile_embolism_only: whether only tiles with embolisms should be
         used
        :param leaf_embolism_only: whether only leaves with embolisms should be
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: combined DataBunch DF and list of image root folder path
        """
        super().get_tile_databunch_df(lseq=self, mseq=mseq,
                                      tile_embolism_only=tile_embolism_only,
                                      leaf_embolism_only=leaf_embolism_only,
                                      csv_name=csv_name)


# *__________________________________ Mask ___________________________________*
class MaskSequence(_CurveSequenceMixin, _ImageSequence):
    """
    A sequence of full size Mask Images
    """

    def __init__(self,
                 mpf_path: str = None,
                 folder_path=None,
                 filename_pattern=None,
                 file_list: List[str] = None,
                 creation_mode: bool = False):
        """
        Instantiates a MaskSequence object.

        :param mpf_path: a multipage file path; this used when this object
         is instantiated in creation mode
        :param folder_path: path that contains the image sequence
        :param filename_pattern: the filename pattern of image files
        :param file_list: a list of filenames, this can be used instead of
         providing a folder path and filename pattern
        :param creation_mode: whether the image sequence object should be
         instantiated in creation mode
        """
        # Adds an additional way to create a sequence object - i.e. using a
        # multi-page file
        if mpf_path is not None:
            LOGGER.info(f"Creating a MaskSequence using mpf_path: {mpf_path}")

            self.mpf_path = mpf_path
            _ImageSequence.__init__(self, creation_mode=creation_mode)
        else:
            _ImageSequence.__init__(self, folder_path, filename_pattern,
                                    file_list, creation_mode=creation_mode)

            if self.num_files == 0:
                LOGGER.warning("The file list is empty")

    # *_____________________________ extraction ______________________________*
    def extract_mask_from_multipage(self,
                                    output_path: str,
                                    overwrite: bool = False,
                                    binarise: bool = False) -> None:
        """
        Extracts and saves mask images from a multipage file.

        :param output_path: where the masks should be saved
        :param overwrite: whether images that exist at the same file path
         should be overwritten
        :param binarise: whether the masks should be binarised; i.e 0
         indicates no embolism and 1 indicates embolism
        :return: None
        """
        output_folder_path, output_file_name = output_path.rsplit("/", 1)

        if binarise:
            output_folder_path = output_folder_path + "-binary"

        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        try:
            image_seq = PIL.Image.open(self.mpf_path)
            mask_seq_list = list(PIL.ImageSequence.Iterator(image_seq))
            # The ImageSequence "closes" after it streams so it needs to be
            # "opened" again due to the line above
            image_seq = PIL.Image.open(self.mpf_path)
        except FileNotFoundError as e:
            raise Exception(e, "Please check the mask file path that "
                               "you provided...")

        self.num_files = len(mask_seq_list)

        placeholder_size = floor(log10(self.num_files)) + 1
        with tqdm(total=self.num_files, file=sys.stdout) as pbar:
            for (i, image) in enumerate(PIL.ImageSequence.Iterator(image_seq)):
                final_filename = utilities.create_file_name(output_folder_path,
                                                            output_file_name,
                                                            i,
                                                            placeholder_size)

                self.image_objects.append(Mask(sequence_parent=self.mpf_path))

                self.image_objects[i].create_mask(final_filename, image,
                                                  overwrite, binarise)
                pbar.update(1)

    # *_______________________________ loading _______________________________*
    def load_extracted_images(self,
                              load_image: bool = False,
                              disable_pb: bool = False) -> None:
        """
        Instantiates Mask objects using the file_list attribute and appends
        these objects to the image_objects attribute.

        :param load_image: whether to load the image array belonging to
         Mask being created
        :param disable_pb: whether the progress bar should be disabled
        :return: None
        """
        super().load_extracted_images(Mask, load_image, disable_pb)

    def load_image_array(self, disable_pb: bool = False) -> None:
        """
        Loads all image arrays belonging to the Leaf objects in the sequence.

        :param disable_pb: whether the progress bar should be disabled
        :return: None

        """
        # not strictly necessary but more user friendly
        super().load_image_array(disable_pb)

    def load_tile_sequence(self,
                           load_image: bool = False,
                           folder_path: str = None,
                           filename_pattern: str = None) -> None:
        """
        Loads all tile objects belonging to the Mask objects in the sequence.

        :param load_image: whether the tile arrays should also be loaded
        :param folder_path: the folder path of the tiles
        :param filename_pattern: the filename pattern of the tiles
        :return: None
        """
        # not strictly necessary but more user friendly
        super().load_tile_sequence(load_image, folder_path, filename_pattern)

    # *______________________________ utilities ______________________________*
    def get_databunch_dataframe(self,
                                embolism_only: bool = False,
                                csv_name: str = None) -> \
            Tuple[pd.DataFrame, str]:
        """
        Extracts a databunch dataframe using the images in this sequence. The
        first field is the leaf path and the second field is the mask name.
        This is useful for Fastai. If a csv name is provided the DataFrame
        is saved.

        :param embolism_only: whether only leaves with embolisms should be used
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: DataBunch DF and sequence root folder path
        """
        return super().get_databunch_dataframe(lseq=self.link, mseq=self,
                                               embolism_only=embolism_only,
                                               csv_name=csv_name)

    def get_tile_databunch_df(self,
                              lseq,
                              tile_embolism_only: bool = False,
                              leaf_embolism_only: bool = False,
                              csv_name: str = None)  -> \
            Tuple[pd.DataFrame, List[str]]:
        """
        Extracts a combined databunch df using all tiles belonging to the
        Image objects in the sequence.  The first field is the leaf tile path
        and the second field is the mask tile name. This is useful for Fastai.
        If a csv name is provided the DataFrame is saved.

        :param mseq: a MaskSequence object
        :param tile_embolism_only: whether only tiles with embolisms should be
         used
        :param leaf_embolism_only: whether only leaves with embolisms should be
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: combined DataBunch DF and list of image root folder path
        """
        super().get_tile_databunch_df(lseq=lseq, mseq=self,
                                      tile_embolism_only=tile_embolism_only,
                                      leaf_embolism_only=leaf_embolism_only,
                                      csv_name=csv_name)

    def binarise_sequence(self, disable_pb: bool = False) -> None:
        """
        Binarises all masks in the sequence.

        :param disable_pb: whether the progress bar should be disabled
        :return: None
        """
        with tqdm(total=len(self.image_objects), file=sys.stdout,
                  disable=disable_pb) as pbar:
            for image in self.image_objects:
                image.binarise_self()
                pbar.update(1)


# *================================= Images ==================================*
# *----------------------------- Abstract Class ------------------------------*
class _Image(ABC):
    """
    Abstract Image class
    """

    def __init__(self,
                 path: str = None,
                 sequence_parent: _ImageSequence = None):
        """
        Instantiates an Image object.

        :param path: image file path
        :param sequence_parent: the sequence to which the image belongs
        """
        self.path = path
        self.sequence_parent = sequence_parent

        self.image_array = None
        self.link = None

        self.has_embolism = None
        self.intersection = None
        self.embolism_percent = None
        self.unique_range = np.array([])

    # *__________________________ loading | linking __________________________*
    def load_image(self) -> None:
        """
        Loads the image located at the path attribute. This image is stored
        in the image_array attribute.

        :return: None
        """
        self.image_array = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)

    def link_me(self, image) -> None:
        """
        Links this image to another Image object. The link attribute is
        updated to point to the input image.

        :param image: an Image object
        :return: None
        """
        self.link = image

    # *___________________________ pre-processing ____________________________*
    @abstractmethod
    def binarise_self(self, image: np.array) -> np.array:
        """
        Binarises the input image.

        :param image: an image array
        :return: a binarised image
        """
        return binarise_image(image)

    def trim_image(self,
                   x_size_dir: Optional[Tuple[int, int]] = None,
                   y_size_dir: Optional[Tuple[int, int]] = None,
                   overwrite: bool = True) -> None:
        """
        Trims an image.

        :param y_size_dir: a tuple of (output size, trim_direction), where
         trim direction is either 1 or -1, which indicates to trim from
          either the top or bottom respectively
        :param x_size_dir: a tuple of (output size, trim_direction), where
         trim direction is either 1 or -1, which indicates to trim from
         either the left or right respectively
        :param overwrite: whether images that exist at the same file path
         should be overwritten
        :return: None
        """
        LOGGER.debug(f"Trimming image {self.path}")

        if x_size_dir:
            if not isinstance(x_size_dir, tuple):
                raise ValueError("please provide a tuple input for "
                                 "x_size_dir")
            elif len(x_size_dir) != 2:
                raise ValueError("please provide exactly an output size and "
                                 "trim direction")

        if y_size_dir:
            if not isinstance(y_size_dir, tuple):
                raise ValueError("please provide a tuple input for "
                                 "y_size_dir")
            elif len(y_size_dir) != 2:
                raise ValueError("please provide exactly an output size and "
                                 "trim direction")

        if x_size_dir:
            self.image_array = utilities.trim_image_array(
                self.image_array, x_size_dir[0], axis="x",
                trim_dir=x_size_dir[1])

        if y_size_dir:
            self.image_array = utilities.trim_image_array(
                self.image_array, y_size_dir[0], axis="y",
                trim_dir=y_size_dir[1])

        if overwrite:
            cv2.imwrite(self.path, self.image_array)

    # *_________________________________ EDA _________________________________*
    @abstractmethod
    def extract_embolism_percent(self,
                                 image: np.array,
                                 embolism_px: int = 255) -> float:
        """
        Updates embolism_percent attribute, which is the percentage of
        pixels with embolisms.

        :param image: an image array
        :param embolism_px: the pixel intensity which indicates a pixel is
         an embolism
        :return: the embolism percent
        """
        self.embolism_percent = (np.count_nonzero(image == embolism_px) /
                                 image.size)
        return self.embolism_percent

    @abstractmethod
    def extract_unique_range(self, image: np.array) -> np.array:
        """
        Update the unique_range attribute, which is a list of the unique
        pixels in the image

        :param image: an image array
        :return: unique range list
        """
        self.unique_range = np.unique(image)

        return self.unique_range

    @abstractmethod
    def extract_intersection(self,
                             image: np.array,
                             combined_image: np.array) -> np.array:
        """
        Calculates the intersection between the current image and all embolisms
        contained in previous images. The intersection attribute is updated
        and the updated combined image is returned.

        :param image: an image array
        :param combined_image: a combined image array to which the image
         should be compared
        :return: an updated combined image
        """
        self.intersection = np.count_nonzero((combined_image == 255) & (
                image == 255))
        self.intersection = (self.intersection / image.size)

        combined_image[image == 255] = 255

        return combined_image

    def extract_has_embolism(self, embolism_px: int = 255) -> None:
        """
        Updates the has_embolism attribute, which is a boolean that
        indicates whether the current image has any embolisms.

        :param embolism_px: the pixel intensity which indicates a pixel is
         an embolism
        :return: None
        """
        if self.embolism_percent > 0:
            self.has_embolism = True
        elif embolism_px in self.unique_range:
            self.has_embolism = True
        else:
            self.has_embolism = False

    # *______________________________ utilities ______________________________*
    def show(self) -> None:
        """
        Displays the image array attribute of the Image object.

        :return: None
        """
        if self.image_array is not None:
            plt.imshow(self.image_array, cmap="gray")
            plt.show()
        else:
            raise Exception("Please load the image first")

    def __str__(self):
        return f"This object is a {self.__class__.__name__}"


# *--------------------- Common Function Implementation ----------------------*
# *__________________________________ Leaf ___________________________________*
class _LeafImage(_Image):
    """
    Contains implementations of abstract functions from _Image that apply to
    images of leaves, these functions are common between both full size
    images and tiles
    """

    def __init__(self,
                 path: str = None,
                 sequence_parent: LeafSequence = None):
        """
        Instantiates a Leaf object.

        :param path: leaf image file path
        :param sequence_parent: the LeafSequence to which this leaf belongs
        """
        super().__init__(path, sequence_parent)
        self.prediction_array = np.array([])

    # *___________________________ pre-processing ____________________________*
    def binarise_self(self, prediction: bool = False) -> None:
        """
        Binarises the leaf image array. Either the prediction array or the
        leaf image array can be used.

        :param prediction: whether the prediction array should be binarised.
        :return: None
        """
        if prediction:
            self.prediction_array = super().binarise_self(
                self.prediction_array)
        else:
            self.image_array = super().binarise_self(self.image_array)

    # *_________________________________ EDA _________________________________*
    def extract_embolism_percent(self,
                                 prediction: bool = False,
                                 embolism_px: int = 255) -> float:
        """
        Updates embolism_percent attribute, which is the percentage of
        pixels with embolisms.

        :param prediction: whether the prediction array should be binarised
        :param embolism_px: the pixel intensity which indicates a pixel is
         an embolism
        :return: None
        """
        if prediction:
            return super().extract_embolism_percent(self.prediction_array,
                                                    embolism_px)
        else:
            return super().extract_embolism_percent(self.image_array,
                                                    embolism_px)

    def extract_unique_range(self, prediction: bool = False) -> np.array:
        """
        Update the unique_range attribute, which is a list of the unique
        pixels in the image

        :param prediction: whether the prediction array should be binarised
        :return: unique range list
        """
        if prediction:
            return super().extract_unique_range(self.prediction_array)
        else:
            return super().extract_unique_range(self.image_array)

    def extract_intersection(self,
                             combined_image: np.array,
                             prediction: bool = False) -> np.array:
        """
        Calculates the intersection between the current image and all embolisms
        contained in previous images. The intersection attribute is updated
        and the updated combined image is returned.

        :param combined_image: a combined image array to which the image
         should be compared
        :param prediction: whether the prediction array should be binarised
        :return: updated combined image
        """
        if prediction:
            return super().extract_intersection(self.prediction_array,
                                                combined_image)
        else:
            return super().extract_intersection(self.image_array,
                                                combined_image)

    def load_image(self,
                   shift_256: bool = False,
                   transform_uint8: bool = False) -> None:
        """
        Loads the image located at the path attribute. This image is stored
        in the image_array attribute.

        :param shift_256: whether images should be shifted by 256
        :param transform_uint8: whether images transformed to a uint8 format
        :return: None
        """
        # default is uint8, since this is usually how images are displayed
        super(_LeafImage, self).load_image()

        if shift_256 and transform_uint8:
            LOGGER.warning("Both shift_256 and transform_uint8 were set to "
                           "true. The shift_256 parameter will be used.")
        # shift 256 will take preference since it's default is false
        if shift_256:
            # if the image was shifted by 256 when saved, then shift back to
            # restore negative values
            self.image_array = self.image_array.astype(np.int16) - 256
        elif transform_uint8:
            # if a shifted image was provided convert back to a uint8 to view
            # note, can't convert back
            self.image_array = self.image_array.astype(np.uint8)


# *__________________________________ Mask ___________________________________*
class _MaskImage(_Image):
    """
    Contains implementations of abstract functions from _Image that apply to
    images of masks, these functions are common between both full size
    images and tiles
    """

    def __init__(self,
                 path: str = None,
                 sequence_parent: MaskSequence = None):
        """
        Instantiates a Mask object.

        :param path: mask image file path
        :param sequence_parent: the MaskSequence to which this leaf belongs
        """
        super().__init__(path, sequence_parent)

    # *___________________________ pre-processing ____________________________*
    def binarise_self(self) -> None:
        """
        Binarises the leaf image array.

        :return: None
        """
        self.image_array = super().binarise_self(self.image_array)

    # *_________________________________ EDA _________________________________*
    def extract_embolism_percent(self, embolism_px: int = 255) -> float:
        """
        Updates embolism_percent attribute, which is the percentage of
        pixels with embolisms.

        :param embolism_px: the pixel intensity which indicates a pixel is
         an embolism
        :return: None
        """
        super().extract_embolism_percent(self.image_array, embolism_px)

    def extract_unique_range(self) -> np.array:
        """
        Update the unique_range attribute, which is a list of the unique
        pixels in the image

        :return: unique range list
        """
        super().extract_unique_range(self.image_array)

    def extract_intersection(self, combined_image: np.array) -> np.array:
        """
        Calculates the intersection between the current image and all embolisms
        contained in previous images. The intersection attribute is updated
        and the updated combined image is returned.

        :param combined_image: a combined image array to which the image
         should be compared
        :return: updated combined image
        """
        return super().extract_intersection(self.image_array, combined_image)


# *---------------------------------- Mixin ----------------------------------*
class _FullImageMixin:
    """
    Allows a full leaf to be split into tiles and load a sequence of Tiles,
    the functions add to both _Image and _ImageSequence functionality
    """

    # *_______________________________ tiling ________________________________*
    def tile_me(self,
                TileClass,
                length_x: int,
                stride_x: int,
                length_y: int,
                stride_y: int,
                output_path: str = None,
                overwrite: bool = False) -> None:
        """
        Tiles an image and creates TileClass objects. These are appended to
        the image_object attribute.

        :param TileClass: tile class to instantiate using the details of the
         new tile created
        :param length_x: the x-length of the tile
        :param stride_x: the size of the x stride
        :param length_y: the y-length of the tile
        :param stride_y: the size of the y stride
        :param output_path: output path of where the tiles should be saved;
         if no path is  provided, tiles are saved in a default location
        :param overwrite: whether tiles that exist at the same file path should
         be overwritten
        :return: None
        """

        if output_path is None:
            output_folder_path, _, output_file_name = self.path.rsplit("/",
                                                                       2)
            output_folder_path, _, output_file_name = self.path.rsplit("/", 2)
            output_folder_path = (output_folder_path + "/chips-" +
                                  self.__class__.__name__.lower())
        else:
            output_folder_path, output_file_name = output_path.rsplit("/",
                                                                      1)

        input_y_length = self.image_array.shape[0]  # rows = y
        input_x_length = self.image_array.shape[1]  # cols = x

        counter = 0

        x_num_tiles = ceil((input_x_length - length_x) / stride_x) + 1
        y_num_tiles = ceil((input_y_length - length_y) / stride_y) + 1
        num_tiles = x_num_tiles * y_num_tiles
        placeholder_size = floor(log10(num_tiles)) + 1

        for y_range in chip_range(0, input_y_length, length_y, stride_y):
            for x_range in chip_range(0, input_x_length, length_x,
                                      stride_x):
                final_filename = utilities.create_file_name(
                    output_folder_path, output_file_name, counter,
                    placeholder_size)

                self.image_objects.append(TileClass(sequence_parent=self,
                                                    path=final_filename))
                self.image_objects[counter].create_tile(
                    length_x, length_y, x_range, y_range, final_filename,
                    overwrite)

                counter += 1

    def load_tile_paths(self,
                        folder_path: str = None,
                        filename_pattern: str = None) -> None:
        """
        Loads all tile objects belonging to the Image.

        :param load_image: whether the tile arrays should also be loaded
        :param folder_path: the folder path of the tiles
        :param filename_pattern: the filename pattern of the tiles
        :return:
        """
        if folder_path is None and filename_pattern is None:
            folder_path, _, filename_pattern = self.path.rsplit(
                "/", 2)
            if isinstance(self, Mask):
                folder_name = "/chips-mask"
            else:
                folder_name = "/chips-leaf"

            folder_path = folder_path + folder_name
            filename_pattern = filename_pattern.rsplit(".")[0] + "*"

        self.file_list = sorted([
            f for f in glob(folder_path + "/" + filename_pattern,
                            recursive=True)])

        self.num_files = len(self.file_list)


# *----------------------------- Implementation ------------------------------*
# *__________________________________ Leaf ___________________________________*
class Leaf(_FullImageMixin, _LeafImage, _ImageSequence):
    """
    A full Leaf Image
    """

    def __init__(self,
                 path=None,
                 sequence_parent=None,
                 parents=None,
                 folder_path=None,
                 filename_pattern=None,
                 file_list: List[str] = None):
        """
        Instantiates a Leaf object.

        :param path: image file path
        :param sequence_parent: the LeafSequence to which the Leaf belongs
        :param parents: the paths to the two files from which this Leaf was
         created
        :param folder_path: the folder path of the tiles belonging to this
         image; this can be left blank unless tiles are also being loaded
        :param filename_pattern: the filename pattern of the tiles;
         this can be left blank unless tiles are also being loaded
        :param file_list: a file list of tile paths;
         this can be used instead of folder_path and filename pattern,
         but it can be left blank unless tiles are also being loaded
        """
        # Can create a Leaf using parents or path
        # Issue with using super is passing the arguments ... could find a
        # way to use kwargs
        _LeafImage.__init__(self, path, sequence_parent)
        _ImageSequence.__init__(self, folder_path, filename_pattern,
                                file_list)

        # refers to the paths of the original leaf images
        if parents is not None:
            self.parents = parents

    # *_____________________________ extraction ______________________________*
    def extract_me(self,
                   filepath: os.path,
                   combination_function=ImageChops.subtract_modulo,
                   shift_256=False,
                   overwrite: bool = False) -> None:
        """
        Extracts and saves changed leaf images. The extracted image and file
        path are stored in the image_array and path attributes
        respectively

        :param filepath: the filepath to save the extracted image
        :param combination_function: the combination function to apply to
         images parents
        :param shift_256: whether the extracted image should be shifted by 256
        :param overwrite: whether an image that exist at the same file path
         should be overwritten
        :return: None
        """
        try:
            old_image = PIL.Image.open(self.parents[0])
            new_image = PIL.Image.open(self.parents[1])
        except FileNotFoundError as e:
            raise Exception(e, "Please check the parent file paths that "
                               "you provided...")

        if shift_256:
            # shift the image so that the full subtraction range is preserved
            # i.e. no wrapping due to using uint8
            combined_image = (np.array(new_image).astype(np.int16) -
                              np.array(old_image).astype(np.int16) + 256)
            combined_image = PIL.Image.fromarray(combined_image)
        else:
            combined_image = combination_function(new_image, old_image)

        create_file = False

        if not os.path.exists(filepath):
            create_file = True

        if overwrite:
            create_file = True

        if create_file:
            LOGGER.debug(f"Creating File: {filepath}")
            with open(filepath, "w") as f:
                combined_image.save(filepath)

        self.image_array = np.array(combined_image)
        self.path = filepath

    # *__________________________ loading | linking __________________________*
    def load_extracted_images(self,
                              load_image: bool = False,
                              disable_pb: bool = False,
                              shift_256: bool = False,
                              transform_uint8: bool = False) -> None:
        """
        Loads LeafTiles belonging to the Leaf.

        :param load_image: whether to load the image array belonging to
         LeafTile being created
        :param disable_pb: whether the progress bar should be disabled
        :param shift_256: whether images should be shifted by 256; applies
         if load_image is true
        :param transform_uint8: whether images transformed to a uint8 format;
         applies if load_image is true
        :return: None
        """
        _ImageSequence.load_extracted_images(self, LeafTile, load_image,
                                             disable_pb, shift_256=shift_256,
                                             transform_uint8=transform_uint8)

    # *_______________________________ tiling ________________________________*
    def tile_me(self,
                length_x: int,
                stride_x: int,
                length_y: int,
                stride_y: int,
                output_path: str = None,
                overwrite: bool = False) -> None:
        """
        Tiles an image and creates LeafTile objects. These are appended to
        the image_object attribute.

        :param length_x: the x-length of the tile
        :param stride_x: the size of the x stride
        :param length_y: the y-length of the tile
        :param stride_y: the size of the y stride
        :param output_path: output path of where the tiles should be saved;
         if no path is  provided, tiles are saved in a default location
        :param overwrite: whether tiles that exist at the same file path should
         be overwritten
        :return: None
        """
        super().tile_me(LeafTile, length_x, stride_x, length_y, stride_y,
                        output_path, overwrite)

    # *_____________________________ prediction ______________________________*
    def predict_leaf(self,
                     model,
                     x_tile_length: int = None,
                     y_tile_length: int = None,
                     memory_saving: bool = True,
                     overwrite: bool = False,
                     save_prediction: bool = True,
                     shift_256: bool = False,
                     transform_uint8: bool = False,
                     threshold: float = 0.5, **kwargs) -> None:
        """
        Predict segmentation maps using the Leaf objects image_array. The
        model used should implement a predict tile method. If memory saving
        is set to false a prediction array is assigned to the Leaf object.

        :param model: a model which inherits Model and hence implements a
         predict tile method
        :param x_tile_length: the x length of the tile used in the original
         training
        :param y_tile_length: the y length of the tile used in the original
         training
        :param memory_saving: if set to True, both the image array and
         prediction array are set to None; this should only be set to true
         if the predictions are being saved
        :param overwrite: whether images that exist at the same file path
         should be overwritten
        :param save_prediction: whether the prediction should be saved
        :param shift_256: whether images should be shifted by 256
        :param transform_uint8: whether images transformed to a uint8 format
        :param threshold: the threshold to use when saving predictions; i.e. a
         pixel is saved as an embolism if p(embolism) > threshold
        :param kwargs: kwargs for the predict tile function
        :return: None
        """

        if self.image_array is None:
            self.load_image(shift_256=shift_256,
                            transform_uint8=transform_uint8)

        counter = 0
        y_length, x_length = self.image_array.shape

        self.prediction_array = np.zeros((y_length, x_length))

        if x_tile_length is None or y_tile_length is None:
            y_tile_length, x_tile_length = self.image_objects[0].shape

        old_upper_y = 0

        for y_range in chip_range(0, y_length, y_tile_length, y_tile_length):
            old_upper_x = 0

            for x_range in chip_range(0, x_length, x_tile_length,
                                      x_tile_length):
                temp_tile = LeafTile(sequence_parent=self)
                temp_tile.create_tile(x_tile_length, y_tile_length,
                                      x_range, y_range)

                pred_tile = temp_tile.predict_tile(model,
                                                   **kwargs)

                pred_tile = pred_tile.reshape(y_tile_length, x_tile_length)

                if ((y_range[1] - old_upper_y) != y_tile_length or
                        (x_range[1] - old_upper_x) != x_tile_length):
                    pred_tile = pred_tile[
                                (old_upper_y - y_range[0]):y_range[1],
                                (old_upper_x - x_range[0]):x_range[1]]
                    self.prediction_array[old_upper_y:y_range[1],
                    old_upper_x:x_range[1]] = pred_tile
                else:
                    self.prediction_array[y_range[0]:y_range[1],
                    x_range[0]:x_range[1]] = pred_tile

                old_upper_x = x_range[1]
                counter += 1
            old_upper_y = y_range[1]

        if save_prediction:
            folder_path, filename = self.path.rsplit("/", 1)
            filename = "pred_" + filename.rsplit(".", 1)[0] + ".png"
            output_folder_path = os.path.join(folder_path, "../predictions")
            filepath = os.path.join(output_folder_path, filename)
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)

            create_file = False

            if not os.path.exists(filepath):
                create_file = True

            if overwrite:
                create_file = True

            if create_file:
                temp_pred = self.prediction_array.copy()
                temp_pred[temp_pred < threshold] = 0
                temp_pred[temp_pred >= threshold] = 255

                cv2.imwrite(filepath, temp_pred)

        if memory_saving:
            self.image_array = None
            self.prediction_array = None

    # *______________________________ utilities ______________________________*
    def get_databunch_dataframe(self,
                                embolism_only: bool = False,
                                csv_name: str = None) -> \
            Tuple[pd.DataFrame, str]:
        """
        Extracts a databunch dataframe using the tiles in this Leaf. The
        first field is the leaf tile path and the second field is the mask
        tile name. This is useful for Fastai. If a csv name is provided the
        DataFrame is saved.

        :param embolism_only: whether only leaves with embolisms should be used
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: DataBunch DF and sequence root folder path
        """
        return super().get_databunch_dataframe(lseq=self,
                                               mseq=self.link,
                                               embolism_only=embolism_only,
                                               csv_name=csv_name)


# *__________________________________ Mask ___________________________________*
class Mask(_FullImageMixin, _MaskImage, _ImageSequence):
    """
    A full Mask Image
    """

    def __init__(self,
                 path: str = None,
                 sequence_parent: MaskSequence = None,
                 folder_path: str = None,
                 filename_pattern: str = None,
                 file_list: List[str] = None):
        """
        Instantiates a Mask object.

        :param path: image file path
        :param sequence_parent: the MaskSequence to which the Mask belongs
        :param folder_path: the folder path of the tiles belonging to this
         image; this can be left blank unless tiles are also being loaded
        :param filename_pattern: the filename pattern of the tiles;
         this can be left blank unless tiles are also being loaded
        :param file_list: a file list of tile paths;
         this can be used instead of folder_path and filename pattern,
         but it can be left blank unless tiles are also being loaded
        """
        _MaskImage.__init__(self, path, sequence_parent)
        _ImageSequence.__init__(self, folder_path, filename_pattern,
                                file_list)

    # *_____________________________ extraction ______________________________*
    def create_mask(self,
                    filepath: Union[Path, str],
                    image,
                    overwrite: bool = False,
                    binarise: bool = False) -> None:
        """
        Saves the PIL image at the provided file path. The  image and file
        path are stored in the image_array and path attributes
        respectively.

        :param filepath: the filepath to save the extracted image (as a
         Path, or string)
        :param image: the mask image (as a PIL image)
        :param overwrite: whether an image that exist at the same file path
         should be overwritten
        :param binarise: whether the mask should be binarised; this assumes
         that embolisms are indicated by a pixel intensity of 255
        :return: None
        """
        self.image_array = np.array(image)

        create_file = False

        if not os.path.exists(filepath):
            create_file = True

        if overwrite:
            create_file = True

        if create_file:
            LOGGER.debug(f"Creating File: {filepath}")

            if binarise:
                self.image_array = self.image_array / 255

            # using plt since image has already been converted to an array
            cv2.imwrite(filepath, self.image_array)

        self.path = filepath

    # *__________________________ loading | linking __________________________*
    def load_extracted_images(self,
                              load_image: bool = False,
                              disable_pb: bool = False) -> None:
        """
        Loads MaskTiles belonging to the Mask.

        :param load_image: whether to load the image array belonging to
         LeafTile being created
        :param disable_pb: whether the progress bar should be disabled
        :return: None
        """
        _ImageSequence.load_extracted_images(self, MaskTile, load_image,
                                             disable_pb)

    # *_______________________________ tiling ________________________________*
    def tile_me(self,
                length_x: int,
                stride_x: int,
                length_y: int,
                stride_y: int,
                output_path: str = None,
                overwrite: bool = False) -> None:
        """
        Tiles an image and creates MaskTile objects. These are appended to
        the image_object attribute.

        :param length_x: the x-length of the tile
        :param stride_x: the size of the x stride
        :param length_y: the y-length of the tile
        :param stride_y: the size of the y stride
        :param output_path: output path of where the tiles should be saved;
         if no path is  provided, tiles are saved in a default location
        :param overwrite: whether tiles that exist at the same file path should
         be overwritten
        :return: None
        """
        super().tile_me(MaskTile, length_x, stride_x, length_y, stride_y,
                        output_path)

    # *______________________________ utilities ______________________________*
    def get_databunch_dataframe(self,
                                embolism_only: bool = False,
                                csv_name: str = None) -> \
            Tuple[pd.DataFrame, str]:
        """
        Extracts a databunch dataframe using the tiles in this Mask. The
        first field is the leaf tile path and the second field is the mask
        tile name. This is useful for Fastai. If a csv name is provided the
        DataFrame is saved.

        :param embolism_only: whether only leaves with embolisms should be used
        :param csv_name: the name of the csv, which can also be a path; if
         this not provided, the DF will not be save
        :return: DataBunch DF and sequence root folder path
        """
        return super().get_databunch_dataframe(lseq=self.link, mseq=self,
                                               embolism_only=embolism_only,
                                               csv_name=csv_name)


# *================================== Tiles ==================================*
# *---------------------------------- Mixin ----------------------------------*
class _TileMixin:
    """
    Adds the ability for a Tile object to create a tile from it's parent's
    image array
    """

    def __init__(self):
        """
        Instantiates a _TileMixin, which adds the padded attribute to a Tile
        """
        self.padded = False

    def create_tile(self,
                    length_x: int,
                    length_y: int,
                    x_range: Tuple[int, int],
                    y_range: Tuple[int, int],
                    filepath: str = None,
                    overwrite: bool = False) -> None:
        """
        Creates a tile by chipping the image_array of the tile's parent object.

        :param length_x: the x-length of the tile
        :param length_y: the y-length of the tile
        :param x_range: the x range of the parent object's image array to chip
        :param y_range: the y range of the parent object's image array to chip
        :param filepath: output path of where the tile should be saved
        :param overwrite: whether a tiles that exists at the same file path
         should be overwritten
        :return: None
        """

        image_chip = chip_image(self.sequence_parent.image_array,
                                x_range, y_range)

        ychip_length = image_chip.shape[0]  # rows = y
        xchip_length = image_chip.shape[1]  # cols = x

        LOGGER.debug(f"{x_range} \t {y_range}")

        if ychip_length < length_y or xchip_length < length_x:
            # TODO: Fix duplication of this if statements due to
            #  lack of identifiability in the above if statement
            image_chip = pad_chip(image_chip, length_x, length_y)
            self.padded = True

        if filepath is not None:
            create_file = False

            if not os.path.exists(filepath):
                create_file = True

            if overwrite:
                create_file = True

            if create_file:
                cv2.imwrite(filepath, image_chip)

        self.image_array = image_chip


# *----------------------------- Implementation ------------------------------*
# *__________________________________ Mask ___________________________________*
class MaskTile(_TileMixin, _MaskImage):
    """
    A Mask tile
    """

    def __init__(self, path=None, sequence_parent=None):
        """
        Instantiates a MaskTile object.

        :param path: tile image filepath
        :param sequence_parent: the Mask object to which this tile belongs
        """
        _MaskImage.__init__(self, path, sequence_parent)
        _TileMixin.__init__(self)


# *__________________________________ Leaf ___________________________________*
class LeafTile(_TileMixin, _LeafImage):
    """
    A Leaf tile
    """

    def __init__(self, path=None, sequence_parent=None):
        """
        Instantiates a LeafTile object.

        :param path: tile image filepath
        :param sequence_parent: the Leaf object to which this tile belongs
        """
        _LeafImage.__init__(self, path, sequence_parent)
        _TileMixin.__init__(self)

    # *_____________________________ prediction ______________________________*
    def predict_tile(self,
                     model: Model,
                     memory_saving: bool = True,
                     **kwargs) -> np.array:
        """
        Predicts and returns a segmentation map using the tile image.

        :param model: a model which inherits Model and hence implements a
         predict tile method
        :param memory_saving: if set to True, the prediction array is not saved
        :param kwargs: kwargs for the predict tile function
        :return: the prediction
        """
        # Accommodates for batch size > 1... need to update for case when
        # batch size = 1? Also fast ai specific
        # input = self.image_array[None, ...]

        prediction_array = model.predict_tile(
            new_tile=self.image_array, **kwargs)

        if not memory_saving:
            self.prediction_array = prediction_array

        return prediction_array
