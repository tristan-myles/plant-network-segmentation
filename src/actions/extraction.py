import logging
from typing import List, Dict

LOGGER = logging.getLogger(__name__)


def extract_leaf_images(lseq_list,
                        output_path_list: List[str],
                        overwrite: bool,
                        format_dict: Dict) -> None:
    """
    Extract differenced leaf images from a list of LeafSequence objects.
    No objects are returned, but the LeafSequence objects will be mutated.

    :param lseq_list: list of LeafSequences
    :param output_path_list: list of file paths of where the differenced
     images should be saved; this path is enumerated as images are saved
    :param overwrite: whether leaves that exist at the same file path should
     be overwritten
    :param format_dict: what format the leaves should be extracted in,
     e.g. should they be shifted by 256
    :return: None
    """
    LOGGER.info("Extracting differenced leaf images")
    for lseq, output_path in zip(lseq_list, output_path_list):
        LOGGER.info(f"Differencing images in {lseq.folder_path} and saving "
                    f"to {output_path}")
        lseq.extract_changed_leaves(output_path, overwrite=overwrite,
                                    **format_dict)


def extract_multipage_mask_images(mseq_list,
                                  output_path_list: List[str],
                                  overwrite: bool,
                                  binarise: bool) -> None:
    """
    Extract mask images from a multipage file.
    No objects are returned, but the MaskSequence objects will be mutated.

    :param mseq_list: list of MaskSequence objects
    :param output_path_list: list of file paths of where the mask
     images should be saved; this path is enumerated as images are saved
    :param overwrite: whether masks that exist at the same file path should
     be overwritten
    :param binarise: whether the masks should be saved in a binary format;
    i.e. 0 for no-embolism and 1 for embolism
    :return: None
    """
    LOGGER.info("Extracting mask images from multipage file")

    for mseq, output_path in zip(mseq_list, output_path_list):
        LOGGER.info(f"Extracting images from: {mseq.mpf_path} and saving "
                    f"to {output_path}")
        mseq.extract_mask_from_multipage(output_path,
                                         overwrite,
                                         binarise)

        # frees up ram when extracting many sequences
        mseq.unload_extracted_images()


def extract_tiles(seq_objects,
                  length_x: int,
                  stride_x: int,
                  length_y: int,
                  stride_y: int,
                  output_path_list: List[str] = None,
                  overwrite: bool = False,
                  **kwargs):
    """
    Extract tiles from either a list of mask or leaf sequence objects. The
    tiles are not stored in the Sequence objects to save memory, hence the
    sequence images are not mutated when they are returned.

    :param seq_objects: list of either MaskSequence or LeafSequence objects
    :param length_x: the x-length of the tile
    :param stride_x: the size of the x stride
    :param length_y: the y-length of the tile
    :param stride_y: the size of the y stride
    :param output_path_list: list of file paths of where the mask
     images should be saved; if no path is provided, tiles are saved in a
     default location
    :param overwrite: whether tiles that exist at the same file path should
     be overwritten
    :param kwargs: kwargs for how the sequence object images should be loaded
    :return: None
    """
    LOGGER.info(f"Extracting tiles from {seq_objects[0].__class__.__name__} "
                f"with the following configuration:"
                f" length (x, y): ({length_x}, {length_y}) |"
                f" stride (x, y): ({stride_x}, {stride_y})")

    if output_path_list is None:
        output_path_list = [None for _ in range(len(seq_objects))]

    for i, seq in enumerate(seq_objects):
        seq.load_image_array(**kwargs)

        seq.tile_sequence(length_x=length_x, stride_x=stride_x,
                          length_y=length_y, stride_y=stride_y,
                          output_path=output_path_list[i],
                          overwrite=overwrite)

        seq.unload_extracted_images()


