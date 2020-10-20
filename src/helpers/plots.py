import logging
import tqdm
import sys

from src.eda.describe_leaf import plot_embolisms_per_leaf

LOGGER = logging.getLogger(__name__)


# *============================ embolism profile =============================*
def plot_mseq_profiles(mseqs, show, output_path_list, leaf_name_list):
    for i, mseq in enumerate(mseqs):
        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images(load_image=True)

        LOGGER.info("Extracting the intersection list")
        mseq.get_intersection_list()

        LOGGER.info("Extracting the embolism percent list")
        mseq.get_embolism_percent_list()

        if output_path_list is not None:
            mseq.plot_profile(show=show, output_path=output_path_list[i],
                              leaf_name=leaf_name_list[i], figsize=(10, 15))
        else:
            mseq.plot_profile(show=show, leaf_name=leaf_name_list[i],
                              figsize=(10, 15))

        mseq.unload_extracted_images()


# *============================ embolism bar plot ============================*
def plot_mseq_embolism_counts(mseqs, show, output_path, tiles,
                              leaf_name_list, leaf_embolism_only=False,
                              percent=False):
    has_embolism_lol = []

    for i, mseq in enumerate(mseqs):
        has_embolism_lol.append([])

        # less memory intensive for images to be loaded here
        LOGGER.info(f"Creating {mseq.num_files} image objects for "
                    f"{mseq.__class__.__name__} located at {mseq.folder_path}")
        mseq.load_extracted_images()

        if not tiles or leaf_embolism_only:
            mseq.load_image_array()

            LOGGER.info("Extracting the embolism percent list")
            mseq.get_embolism_percent_list()

            LOGGER.info("Extracting the has_embolism list")
            mseq.get_has_embolism_list()

            mseq.unload_extracted_images()

        if not tiles:
            has_embolism_lol[i] = has_embolism_lol[i] + mseq.has_embolism_list

        if tiles:
            with tqdm(total=len(mseq.image_objects), file=sys.stdout) as pbar:
                for mask_image in mseq.image_objects:
                    if leaf_embolism_only:
                        # Assumes linked sequences have been provided.
                        if not mask_image.has_embolism:
                            # Skip this image if the mask has no embolism
                            continue

                    mask_image.load_tile_paths()
                    mask_image.load_extracted_images(load_image=True,
                                                     disable_pb=True)
                    mask_image.get_embolism_percent_list(disable_pb=True)
                    mask_image.get_has_embolism_list(disable_pb=True)
                    has_embolism_lol[i] = \
                        has_embolism_lol[i] + mask_image.has_embolism_list
                    mask_image.unload_extracted_images()
                    pbar.update(1)

    if output_path is not None:
        plot_embolisms_per_leaf(has_embolism_lol=has_embolism_lol,
                                show=show, output_path=output_path,
                                leaf_names_list=leaf_name_list,
                                percent=percent, figsize=(15, 10))
    else:
        plot_embolisms_per_leaf(has_embolism_lol=has_embolism_lol,
                                show=show, leaf_names_list=leaf_name_list,
                                percent=percent, figsize=(15, 10))

        mseq.unload_extracted_images()
# *===========================================================================*
