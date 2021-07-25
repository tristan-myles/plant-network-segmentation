import logging
from typing import List, Dict

from src.data_model.data_model import LeafSequence, MaskSequence

LOGGER = logging.getLogger(__name__)


# *===========================================================================*
def extract_full_eda_df(mseq_list: List[MaskSequence],
                        options: Dict,
                        output_path_list: List[str],
                        lseq_list: List[LeafSequence] = None) -> None:
    """
    Creates and saves a list of full size image EDA dataframes from a list
    MaskSequences.

    :param mseq_list: A list of MaskSequence objects
    :param options: the options of what should be included in the dataframe;
     the option name should be the key and the value should be either true or
     false
    :param output_path_list: the list of output csv file paths
    :param lseq_list: A list of LeafSequence objects; this is only required if
     the linked filename option is used
    :return: None
    """
    for i, (mseq, csv_output_path) in enumerate(zip(mseq_list,
                                                    output_path_list)):
        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images(load_image=True)

        if options["linked_filename"]:
            mseq.link_sequences(lseq_list[i])

        _ = mseq.get_eda_dataframe(options, csv_output_path)

        mseq.unload_extracted_images()


def extract_tiles_eda_df(mseq_list: List[MaskSequence],
                         options: Dict,
                         output_path_list: List[str],
                         lseq_list: List[LeafSequence] = None) -> None:
    """
    Creates and saves a list of tile image EDA dataframes from a list
    MaskSequences.

    :param mseq_list: A list of MaskSequence objects
    :param options: the options of what should be included in the dataframe;
     the option name should be the key and the value should be either true or
     false
    :param output_path_list: the list of output csv file paths
    :param lseq_list: A list of LeafSequence objects; this is only required if
     the linked filename option is used
    :return: None
    """
    for i, (mseq, csv_output_path) in enumerate(zip(mseq_list, output_path_list)):
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")

        if options["linked_filename"]:
            mseq.link_sequences(lseq_list[i])

        _ = mseq.get_tile_eda_df(options, csv_output_path)

        mseq.unload_extracted_images()


def extract_full_databunch_df(lseq_list:  List[LeafSequence],
                              mseq_list:  List[MaskSequence],
                              output_path_list: List[str],
                              embolism_only=False) -> None:
    """
    Extracts a databunch dataframe of full size images. The first field is
    the leaf path and the second field is the mask name.  This is useful for
    Fastai.

    :param lseq_list: A list of LeafSequence objects
    :param mseq_list: A list of MaskSequence objects
    :param output_path_list: the list of output csv file paths
    :param embolism_only: whether only leaves with embolisms should be used
    :return: None
    """
    for csv_output_path, lseq, mseq in zip(output_path_list,
                                           lseq_list, mseq_list):
        mseq.load_extracted_images(load_image=True)
        mseq.get_embolism_percent_list()
        mseq.get_has_embolism_list()

        lseq.link_sequences(mseq)

        LOGGER.info(f"Creating DataBunch DataFrame using "
                    f"{lseq.__class__.__name__} located at {lseq.folder_path} "
                    f"and {mseq.__class__.__name__} located at"
                    f" {mseq.folder_path}")

        # get_databunch_dataframe written into of a lseq, i.e. will always
        # report lseq regardless of whether it is longer or shorter
        _ = lseq.get_databunch_dataframe(embolism_only, csv_output_path)


def extract_tiles_databunch_df(lseq_list:  List[LeafSequence],
                               mseq_list:  List[MaskSequence],
                               output_path_list: List[str],
                               tile_embolism_only: bool = False,
                               leaf_embolism_only: bool = False) -> None:
    """
    Extracts a databunch dataframe of full size images. The first field is
    the leaf path and the second field is the mask name.  This is useful for
    Fastai.

    :param lseq_list: A list of LeafSequence objects
    :param mseq_list: A list of MaskSequence objects
    :param output_path_list: the list of output csv file paths
    :param tile_embolism_only: whether only tiles with embolisms should be used
    :param leaf_embolism_only: whether only leaves with embolisms should be
     used
    :return: None
    """
    for csv_output_path, lseq, mseq in zip(output_path_list,
                                           lseq_list, mseq_list):
        lseq.link_sequences(mseq)

        LOGGER.info(f"Creating Tile DataBunch DataFrame using "
                    f"{lseq.__class__.__name__} located at {lseq.folder_path} "
                    f"and {mseq.__class__.__name__} located at"
                    f" {mseq.folder_path}")

        _ = lseq.get_tile_databunch_df(mseq, tile_embolism_only,
                                       leaf_embolism_only,
                                       csv_output_path)
# *===========================================================================*
