import PIL.Image
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from PIL import ImageChops
from math import log10, floor
import os

LOGGER = logging.getLogger(__name__)


class ImageSequence:
    # Think of this as a curve
    image_objects = []

    def __init__(self, folder_path=None, filename_pattern=None):
        """
        :param folder_path: path that contains the image sequence
        """
        self.folder_path = folder_path
        self.filename_pattern = filename_pattern

        # original leafs
        self.original_file_list = sorted(
            [f for f in glob(self.folder_path +
                             self.filename_pattern,
                             recursive=True)])

        self.num_files = len(self.original_file_list)

        if self.num_files == 0:
            LOGGER.info("There were no files in that folder...")


class LeafSequence(ImageSequence):
    mask_file_list = []

    def extract_changed_leafs(self,  output_path: str, dif_len: int = 1):
        output_folder_path, output_file_name = output_path.rsplit("/", 1)
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

        if dif_len == 0:
            old_image = self.original_file_list[0]
            step_size = 1
        else:
            step_size = dif_len

        placeholder_size = floor(log10(self.num_files)) + 1

        for i in range(0, self.num_files - step_size):
            if dif_len != 0:
                old_image = self.original_file_list[i]
            new_image = self.original_file_list[i + step_size]

            file_name, extension =\
                str.rsplit(output_file_name, ".",1)
            final_file_name = (f"{file_name}_"
                                f"{i:0{placeholder_size}}.{extension}")
            final_file_name = os.path.join(output_folder_path,
                                            final_file_name)

            self.image_objects.append(
                Leaf(parents=[old_image, new_image]))
            self.image_objects[i].extract_me(final_file_name)


class MaskSequence(ImageSequence):
    def __int__(self, path: str, mpt: bool):
        """

        :param path:
        :param mpt: multi-page tiff boolean
        :return:
        """
        super().__init__(path)
        self.mpt = mpt

    def load_members(self):
        pass


###############################################################################
class Image:
    tiles = None
    embolism_perc = None
    unique_embolism_perc = None
    image_array = None
    parents = []

    def __init__(self, path=None):
        self.path = path

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

    def tile_me(self):
        # Should this be here since a tile inherits this method
        pass

    def extract_embolism_perc(self):
        pass

    def extract_unique_embolism_perc(self):
        pass


class Leaf(Image):
    mask_instance = None

    def __init__(self, path=None, parents=None, mask_instance=None):
    # Can create a Leaf using parents or path
        if path is not None:
            super().__init__(path)
        elif parents is not None:
            self.parents = parents

        if mask_instance is not None:
            self.mask_instance = mask_instance

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
            raise Exception(e, "you gave the wrong filename DUMMY!")

        self.image_array = combination_function(new_image, old_image)

        create_file = False

        if not os.path.exists(filepath):
            create_file = True

        if overwrite:
            create_file = True

        if create_file:
            LOGGER.debug(f"Creating File: {filepath}")
            with open(filepath, "w") as f:
                self.image_array.save(filepath)

        self.image_array = np.array(self.image_array)


class Mask(Image):
    diff_instance = None

    def __init__(self, path, diff_instance=None):
        super().__init__(path)
        if diff_instance is not None:
            self.diff_instance = diff_instance


class Tile(Image):
    def __init__(self, path, parent):
        super().__init__(path)
        self.parent = parent # parent determines the type

    def predict_me(self):
        pass
