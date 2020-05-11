import logging
import os
import sys
from glob import glob
from itertools import chain
from math import log10, floor, ceil
from pathlib import Path
from typing import List, Tuple

import PIL.Image
import PIL.ImageSequence
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageChops
from tqdm import tqdm

from src.eda.describe_leaf import binarise_image
from src.helpers import utilities
from src.helpers.extract_dataset import chip_image, pad_chip, chip_range
from src.eda.describe_leaf import plot_embolism_profile

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class ImageSequence:
    # Think of this as a curve

    def __init__(self, folder_path=None, filename_pattern=None,
                 file_list: List[str] = None, original_images: bool = False):
        """
        :param folder_path: path that contains the image sequence
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

            if self.num_files == 0:
                LOGGER.debug("The file list is empty")

        self.image_objects = []
        self.link = None

        self.unique_range = np.array([])

        self.intersection_list = []
        self.unique_range_list = []
        self.embolism_percent_list = []
        self.linked_path_list = []
        self.has_embolism_list = []

        # two modes - orignal mode | extracted mode:
        self.original_images = original_images

    def load_extracted_images(self, ImageClass,
                              load_image: bool = False):
        if self.original_images:
            raise Exception("The file list contains original images, "
                            "not extracted images. This function is not "
                            "applicable...")

        with tqdm(total=self.num_files, file=sys.stdout) as pbar:
            for i, filename in enumerate(self.file_list):
                self.image_objects.append(ImageClass(filename))
                if load_image:
                    self.image_objects[i].load_image()
                pbar.update(1)

    def unload_extracted_images(self):
        for image in self.image_objects:
            image.image_array = None

    def sort_image_objects_by_filename(self):
        self.image_objects = sorted(self.image_objects,
                                    key=lambda image: image.path)

    def link_sequences(self, SequenceObject, sort_by_filename: bool = True):
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

    def link_tiles(self, sort_by_filename: bool = True):
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                # overwrites link with itself
                image.link_sequences(image.link,
                                     sort_by_filename)
                pbar.update(1)

    def tile_sequence(self, **kwargs):
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                image.tile_me(**kwargs)
                pbar.update(1)

    def load_tile_sequence(self, **kwargs):
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                image.load_tile_paths(**kwargs)
                image.load_extracted_images()
                pbar.update(1)

    def binarise_sequence(self):
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                image.binarise_self()

    def get_unique_sequence_range(self):
        if not self.unique_range_list:
            self.get_unique_range_list()

        self.unique_range = np.unique(list(chain.from_iterable(self.unique_range_list)))

    def get_intersection_list(self):
        self.intersection_list = []

        image_shape = self.image_objects[0].image_array.shape
        # assumes images are the same size
        combined_image = np.empty(shape=(image_shape[0], image_shape[1]),
                                  dtype='object')

        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                combined_image = image.extract_intersection(combined_image)
                self.intersection_list.append(image.intersection)
                pbar.update(1)

    def get_unique_range_list(self):
        self.unique_range_list = []

        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                # no option to overwrite ...
                if image.unique_range.size == 0:
                    image.extract_unique_range()

                    self.unique_range_list.append(image.unique_range)

                pbar.update(1)

    def get_embolism_percent_list(self):
        self.embolism_percent_list = []

        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for image in self.image_objects:
                if image.embolism_percent is None:
                    image.extract_embolism_percent()

                self.embolism_percent_list.append(image.embolism_percent)
                pbar.update(1)

    def get_has_embolism_list(self):
        self.has_embolism_list = []

        with tqdm(total=len(self.image_objects),
                  file=sys.stdout) as pbar:
            for image in self.image_objects:
                if image.has_embolism is None:
                    image.extract_has_embolism()

                self.has_embolism_list.append(image.has_embolism)
                pbar.update(1)

    def get_eda_dataframe(self, options, csv_name: str = None):
        output_dict = {"names": list(map(
            lambda image: image.path.rsplit("/",1)[1], self.image_objects))}

        if options["linked_filename"]:
            # assumes files have been linked
            output_dict["links"] = []
            for image in self.image_objects:
                if image.link is not None:
                    output_dict["links"].append(
                        image.link.path.rsplit("/",1)[1])
                else:
                    output_dict["links"].append("")


        # Unique range
        if options["unique_range"]:
            if not self.unique_range_list:
                self.get_unique_range_list()

            output_dict["unique_range"] = self.unique_range_list

        # Embolism percentages
        if options["embolism_percent"]:
            if not self.embolism_percent_list:
                self.get_embolism_percent_list()

            output_dict["embolism_percent"] = self.embolism_percent_list

        # Extracting intersections
        if options["intersection"]:
            if not self.intersection_list:
                self.get_intersection_list()

            output_dict["intersection"] = self.intersection_list

            # Extracting intersections
        if options["has_embolism"]:
            if not self.has_embolism_list:
                self.get_has_embolism_list()

            output_dict["has_embolism"] = self.has_embolism_list

        output_df = pd.DataFrame(output_dict)
        # Saving the results
        if csv_name:
            output_df.to_csv(csv_name)

        return output_df

    def plot_profile(self, **kwargs):
        plot_embolism_profile(self.embolism_percent_list,
                              self.intersection_list, **kwargs)

    def get_databunch_dataframe(self, lseq, mseq,
                                embolism_only: bool = False,
                                csv_name: str = None):

        output_dict = {"leaf_names":
                           list(map(lambda image: image.path.rsplit("/", 1)[1],
                                    lseq.image_objects)),
                       "masks": []}

        for image in lseq.image_objects:
            if image.link is not None:
                output_dict["masks"].append(image.link.path)
            else:
                output_dict["masks"].append("")

        output_df = pd.DataFrame(output_dict)
        folder_path = lseq.image_objects[0].path.rsplit("/", 1)[0]

        if embolism_only:
            output_df = output_df[mseq.has_embolism_list]

        # Saving the results
        if csv_name:
            output_df.to_csv(csv_name)

        return output_df, folder_path


class LeafSequence(ImageSequence):
    def extract_changed_leaves(self,  output_path: str, dif_len: int = 1,
                               overwrite: bool = False,
                               combination_function=
                               ImageChops.subtract_modulo):
        output_folder_path, output_file_name = output_path.rsplit("/", 1)
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        if dif_len == 0:
            old_image = self.file_list[0]
            step_size = 1
        else:
            step_size = dif_len
        placeholder_size = floor(log10(self.num_files)) + 1

        with tqdm(total=len(self.file_list)-dif_len, file=sys.stdout) as pbar:
            for i in range(0, self.num_files - step_size):
                if dif_len != 0:
                    old_image = self.file_list[i]
                new_image = self.file_list[i + step_size]

                final_filename = utilities.create_file_name(
                    output_folder_path, output_file_name, i, placeholder_size)

                self.image_objects.append(
                    Leaf(parents=[old_image, new_image]))
                self.image_objects[i].extract_me(
                    final_filename,  combination_function, overwrite)
                pbar.update(1)

    def load_extracted_images(self, load_image: bool = False):
        super().load_extracted_images(Leaf, load_image)

    def predict_leaf_sequence(self, model, x_tile_length: int = None,
                     y_tile_length: int = None, memory_saving: bool = True,
                     overwrite: bool = False, save_prediction: bool = True,
                     **kwargs):
        with tqdm(total=len(self.image_objects), file=sys.stdout) as pbar:
            for leaf in self.image_objects:
                leaf.predict_leaf(model, x_tile_length, y_tile_length,
                                 memory_saving, overwrite, save_prediction,
                                  **kwargs)
                pbar.update(1)

    def get_databunch_dataframe(self, embolism_only: bool = False,
                                csv_name: str = None):
        return super().get_databunch_dataframe(lseq=self, mseq=self.link,
                                        embolism_only=embolism_only,
                                        csv_name=csv_name)


class MaskSequence(ImageSequence):
    def __init__(self, folder_path=None, filename_pattern=None,
                 mpf_path: str = None,  original_images: bool = False):
        """

        :param path:
        :param mpf: multi-page file boolean
        :return:
        """
        # Two modes either initialised with a multi-page tiff mask or a
        # folder to a sequence of masks
        if mpf_path is not None:
            self.mpf_path = mpf_path
            super().__init__(original_images=original_images)
        else:
            super().__init__(folder_path, filename_pattern,
                             original_images=original_images)

    def extract_mask_from_multipage(self,  output_path: str,
                                    overwrite: bool = False):
        output_folder_path, output_file_name = output_path.rsplit("/", 1)

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
                                                            i, placeholder_size)

                self.image_objects.append(Mask(sequence_parent=self.mpf_path))

                self.image_objects[i].create_mask(final_filename, image,
                                                  overwrite)
                pbar.update(1)

    def load_extracted_images(self, load_image: bool = False):
        super().load_extracted_images(Mask, load_image)

    def get_databunch_dataframe(self, embolism_only: bool = False,
                                csv_name: str = None):
        return super().get_databunch_dataframe(lseq=self.link, mseq=self,
                                        embolism_only=embolism_only,
                                        csv_name=csv_name)


###############################################################################
class Image(ImageSequence):
    # Should this have a link to a sequence_parent?
    embolism_percent = None
    unique_range = None
    image_array = None
    parents = []
    link = None

    def __init__(self, path=None, file_list: List[str] = None):
        self.path = path
        self.image_objects = []
        self.has_embolism = None
        self.intersection = None

        super().__init__(file_list=file_list)

    def __str__(self):
        return f"This object is a {self.__class__.__name__}"

    def load_image(self):
        self.image_array = np.array(PIL.Image.open(self.path))

    def show(self):
        if self.image_array is not None:
            plt.imshow(self.image_array, cmap="gray")
            plt.show()
        else:
            LOGGER.info("Please load the image first")

    def tile_me(self, TileClass, length_x: int, stride_x: int, length_y: int,
                stride_y: int, output_path: str = None):

        if output_path is None:
            output_folder_path, output_file_name = self.path.rsplit("/", 1)
            output_folder_path = os.path.join(
                output_folder_path,
                "../chips-" + str.lower(self.__class__.__name__))
        else:
            output_folder_path, output_file_name = output_path.rsplit("/", 1)

        if self.image_array is None:
            self.load_image()

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

                self.image_objects.append(TileClass(parent=self))
                self.image_objects[counter].create_tile(
                    length_x, length_y, x_range, y_range, final_filename)

                counter += 1

    def link_me(self, image):
        self.link = image

    def load_tile_paths(self, folder_path: str = None,
                        filename_pattern: str = None):

        if folder_path is None and filename_pattern is None:
            folder_path, filename_pattern = self.path.rsplit("/", 1)
            folder_path = os.path.join(
                folder_path, "../chips-" + str.lower(self.__class__.__name__))
            filename_pattern = filename_pattern.rsplit(".")[0]+"*"

        self.file_list = sorted([f for f in glob(
            folder_path + "/" + filename_pattern, recursive=True)])

        self.num_files = len(self.file_list)

    def load_extracted_images(self, load_image: bool = False):
        super().load_extracted_images(Tile)

    def extract_embolism_percent(self, image: np.array):
        self.embolism_percent = np.count_nonzero(image == 255) / image.size

        return self.embolism_percent

    def extract_has_embolism(self, embolism_px: int = 255):
        if self.embolism_percent > 0:
            self.has_embolism = True
        elif embolism_px in self.unique_range:
            self.has_embolism = True
        else:
            self.has_embolism = False

    def extract_unique_range(self, image: np.array):
        self.unique_range = np.unique(image)

        return self.unique_range

    def extract_intersection(self, image: np.array, combined_image: np.array):
        """
        Calculates the intersection between the current mask and all embolisms
        contained in previous masks
        :param image: np.array of a mask
        :param combined_image: np.array of a combined mask
        :return: the intersection as a % of the image size and an updated
        combined image
        """
        self.intersection = np.count_nonzero((combined_image == 255) & (
                image == 255))
        self.intersection = (self.intersection / image.size)

        combined_image[image == 255] = 255

        return combined_image


class Leaf(Image):
    def __init__(self, path=None, parents=None, mask_instance=None):
        # Can create a Leaf using parents or path
        if path is not None:
            super().__init__(path)
        if parents is not None:
            self.parents = parents

        if mask_instance is not None:
            self.mask_instance = mask_instance

        self.prediction_array = np.array([])

    def link_mask(self, mask_instance):
        self.mask_instance = mask_instance

    def predict_me(self):
        pass

    def extract_me(self, filepath: os.path,
                   combination_function=ImageChops.subtract_modulo,
                   overwrite: bool = False):
        try:
            old_image = PIL.Image.open(self.parents[0])
            new_image = PIL.Image.open(self.parents[1])
        except FileNotFoundError as e:
            raise Exception(e, "Please check the parent file paths that "
                                "you provided...")

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

    def tile_me(self, length_x: int, stride_x: int, length_y: int,
                stride_y: int, output_path: str = None):
        super().tile_me(LeafTile, length_x, stride_x, length_y, stride_y,
                        output_path)

    def extract_embolism_percent(self):
        super().extract_embolism_percent(self.prediction_array)

    def extract_unique_range(self):
        super().extract_unique_range(self.image_array)

    def binarise_self(self, prediction):
        if prediction:
            self.prediction_array = binarise_image(self.prediction_array)
        else:
            self.image_array = binarise_image(self.image_array)

    def extract_intersection(self, combined_image):
        return super().extract_intersection(self.prediction_array,
                                            combined_image)

    def predict_leaf(self, model, x_tile_length: int = None,
                     y_tile_length: int = None, memory_saving: bool = True,
                     overwrite: bool = False, save_prediction: bool = True,
                     **kwargs):

        if self.image_array is None:
            self.load_image()

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
                temp_tile = LeafTile(parent=self)
                temp_tile.create_tile(x_tile_length, y_tile_length,
                                      x_range, y_range)

                pred_tile = temp_tile.predict_tile(model,
                                                   **kwargs)

                pred_tile = (pred_tile[0].px.numpy() * 255)
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
            output_folder_path = os.path.join(folder_path, "predictions")
            filepath = os.path.join(output_folder_path, filename)
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)

            create_file = False

            if not os.path.exists(filepath):
                create_file = True

            if overwrite:
                create_file = True

            if create_file:
                cv2.imwrite(filepath, self.prediction_array)

        if memory_saving:
            self.image_array = None


class Mask(Image):
    def __init__(self, path=None, sequence_parent=None):
        if path is not None:
            super().__init__(path)
        if sequence_parent is not None:
            self.sequence_parent = sequence_parent

    def create_mask(self, filepath: os.path, image: PIL.Image,
                    overwrite: bool = False, binarise: bool = False):
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

    def tile_me(self, length_x: int, stride_x: int, length_y: int,
                stride_y: int, output_path: str = None):
        super().tile_me(Tile, length_x, stride_x, length_y, stride_y,
                        output_path)

    def extract_unique_range(self):
        super().extract_unique_range(self.image_array)

    def extract_embolism_percent(self):
        super().extract_embolism_percent(self.image_array)

    def binarise_self(self):
        self.image_array = binarise_image(self.image_array)

    def extract_intersection(self, combined_image):
        return super().extract_intersection(self.image_array, combined_image)


class Tile(Image):

    def __init__(self, path=None, parent=None):
        if path is not None:
            super().__init__(path)
        if parent is not None:
            self.parent = parent # parent determines the type

        self.padded = False
        self.image_array = None

    def create_tile(self,
                    length_x: int, length_y: int,
                    x_range: Tuple[int, int],
                    y_range: Tuple[int, int],
                    filepath: str = None,
                    overwrite: bool = False,):

        image_chip = chip_image(self.parent.image_array, x_range, y_range)

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


class LeafTile(Tile):
    def __init__(self, path=None, parent=None):
        super().__init__(path, parent)
        self.prediction_array = None

    def predict_tile(self, model, memory_saving: bool = True, **kwargs):
        """

        :param prediction_wrapper: should take in an image_tile
        :return:
        """
        # Accommodates for batch size > 1... need to update for case when
        # batch size = 1? Also fast ai specific
        # input = self.image_array[None, ...]

        prediction_array = model.predict_tile(
            new_tile=self.image_array, **kwargs)

        if not memory_saving:
            self.prediction_array = prediction_array

        return prediction_array
